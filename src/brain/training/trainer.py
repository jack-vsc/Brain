"""Treino supervisionado (cross-entropy + AdamW) do baseline neural.

Garantias:
- perfil de memória antes do treino; em OOM reduz o batch (mantendo o batch
  efetivo via gradient accumulation) antes de qualquer outra coisa; CPU só
  como fallback explícito e registrado;
- precisão mista apenas se um passo de teste funcionar;
- checkpoints atômicos (last.pt e best.pt), retomada com a mesma ordem de
  exemplos, orçamento de tempo e Ctrl+C salvam o estado antes de sair;
- modo piloto: poucos passos para estimar o custo do treino completo.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from brain.data.dataset import AnswerSpace, QADataset, make_collate
from brain.data.generator import load_manifest, load_split
from brain.data.tokenizer import Vocab
from brain.evaluation.metrics import compute_metrics
from brain.evaluation.predict import predict
from brain.models.registry import build_model, count_parameters
from brain.training.checkpoint import (
    capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint,
)
from brain.training.control import PAUSED_EXIT, pause_flag
from brain.training.device import MB, hardware_report, process_ram_mb, select_device

DEFAULTS = {
    "run": {"name": "gru_baseline", "seed": 1, "output_dir": "runs",
            "data_dir": "data/generated/mvp"},
    # Hiperparâmetros de arquitetura vêm dos construtores de cada modelo (não se misturam entre tipos).
    "model": {"type": "gru", "question_position": "after"},
    "train": {
        "batch_size": 64, "grad_accum": 1, "lr": 2e-3, "weight_decay": 0.01,
        "max_steps": 20000, "warmup_steps": 500, "min_lr_ratio": 0.05,
        "clip_norm": 1.0, "eval_every": 500, "log_every": 100,
        "early_stop_patience": 8, "max_minutes": 60.0, "amp": "auto",
        "device": "auto", "min_free_vram_mb": 400, "vram_margin_mb": 256,
        "min_batch_size": 8, "eval_batch_size": 256, "max_train_episodes": 0,
    },
}


def load_config(path: Path | None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    user = tomllib.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    for src in (user, overrides or {}):
        for section, values in src.items():
            cfg.setdefault(section, {}).update(values)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WarmupCosine:
    """Multiplicador de LR: aquecimento linear e decaimento cosseno até min_ratio."""

    def __init__(self, warmup: int, total: int, min_ratio: float):
        self.warmup, self.total, self.min_ratio = warmup, total, min_ratio

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return (step + 1) / max(1, self.warmup)
        t = min(1.0, (step - self.warmup) / max(1, self.total - self.warmup))
        return self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


class EpochSampler(Sampler):
    """Permutação determinística por época, com salto para retomar no meio dela."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch, self.skip = n, seed, 0, 0

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        self.epoch, self.skip = epoch, skip

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        return iter(torch.randperm(self.n, generator=g).tolist()[self.skip:])

    def __len__(self) -> int:
        return self.n - self.skip


class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._log = open(run_dir / "train.log", "a", encoding="utf-8")
        self._metrics = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")

    def info(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def metric(self, record: dict) -> None:
        self._metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._metrics.flush()

    def close(self) -> None:
        self._log.close()
        self._metrics.close()


def _is_oom(err: BaseException) -> bool:
    return isinstance(err, torch.OutOfMemoryError) or "out of memory" in str(err).lower()


def _worst_case_batch(dataset: QADataset, batch_size: int, pad_id: int):
    longest = sorted(range(len(dataset)), key=lambda i: -len(dataset.items[i][0]))[:batch_size]
    return make_collate(pad_id)([dataset[i] for i in longest])


def profile_memory(model, dataset, device: str, batch_size: int, amp: bool,
                   margin_mb: float, min_batch: int, log: RunLogger) -> dict:
    """Mede o pico de memória de um passo com as sequências mais longas.

    Reduz o batch pela metade até caber em (VRAM livre - margem). Se nem o
    batch mínimo couber, recomenda CPU.
    """
    if device != "cuda":
        return {"device": device, "batch_size": batch_size, "peak_mb": None,
                "attempts": [], "note": "sem perfil de VRAM em CPU"}
    attempts = []
    free_mb = torch.cuda.mem_get_info()[0] / MB
    budget = free_mb - margin_mb
    bs = batch_size
    while bs >= min_batch:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated() / MB
        try:
            tokens, lengths, labels, _ = _worst_case_batch(dataset, bs, model.embedding.padding_idx)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = torch.nn.functional.cross_entropy(
                    model(tokens.to(device), lengths), labels.to(device))
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / MB
            reserved = torch.cuda.max_memory_reserved() / MB
            extra = reserved - base
            ok = extra <= budget
            attempts.append({"batch_size": bs, "peak_allocated_mb": round(peak, 1),
                             "peak_reserved_mb": round(reserved, 1), "fits": ok})
            if ok:
                model.zero_grad(set_to_none=True)
                return {"device": "cuda", "batch_size": bs, "peak_mb": round(reserved, 1),
                        "free_vram_mb": round(free_mb), "budget_mb": round(budget),
                        "max_len": int(lengths.max()), "attempts": attempts}
            log.info(f"perfil: batch {bs} usa {extra:.0f} MB > orçamento {budget:.0f} MB; reduzindo")
        except RuntimeError as err:
            if not _is_oom(err):
                raise
            attempts.append({"batch_size": bs, "oom": True})
            log.info(f"perfil: OOM com batch {bs}; reduzindo")
        bs //= 2
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return {"device": "cpu", "batch_size": batch_size, "peak_mb": None,
            "free_vram_mb": round(free_mb), "budget_mb": round(budget), "attempts": attempts,
            "note": f"nem batch {min_batch} coube na VRAM livre; fallback explícito para CPU"}


def amp_works(model, dataset, device: str) -> tuple[bool, str]:
    if device != "cuda":
        return False, "AMP desativado fora da GPU"
    try:
        tokens, lengths, labels, _ = _worst_case_batch(dataset, 4, model.embedding.padding_idx)
        scaler = torch.amp.GradScaler("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(
                model(tokens.to(device), lengths).float(), labels.to(device))
        scaler.scale(loss).backward()
        model.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            return False, "perda não finita no teste de AMP"
        return True, "AMP fp16 testado com sucesso"
    except RuntimeError as err:
        model.zero_grad(set_to_none=True)
        return False, f"AMP falhou no teste: {err}"


def _ckpt_payload(model, optimizer, scheduler, scaler, cfg, state, vocab, answers, manifest,
                  extra: dict | None = None) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng_state(),
        "state": dict(state),
        "config": cfg,
        "vocab": vocab.itos,
        "answer_labels": answers.labels,
        "data_version": manifest["data_version"],
        "generator_version": manifest["generator_version"],
        "torch_version": str(torch.__version__),  # TorchVersion não passa em weights_only
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(extra or {}),
    }


def train(cfg: dict, resume: Path | None = None, pilot_steps: int = 0) -> Path:
    run, mcfg, tcfg = cfg["run"], cfg["model"], cfg["train"]
    seed = int(run["seed"])
    suffix = "-pilot" if pilot_steps else ""
    run_dir = Path(run["output_dir"]) / f"{run['name']}-s{seed}{suffix}"
    log = RunLogger(run_dir)
    set_seed(seed)

    data_dir = Path(run["data_dir"])
    manifest = load_manifest(data_dir)
    vocab = Vocab.load(data_dir / "vocab.json")
    answers = AnswerSpace()
    train_eps = load_split(data_dir, "train")
    if tcfg["max_train_episodes"]:
        train_eps = train_eps[: tcfg["max_train_episodes"]]
    val_eps = load_split(data_dir, "val")
    qpos = mcfg["question_position"]
    train_ds = QADataset(train_eps, vocab, answers, qpos)
    val_ds = QADataset(val_eps, vocab, answers, qpos)

    hw = hardware_report()
    (run_dir / "hardware.json").write_text(json.dumps(hw, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    log.info(f"dados {data_dir} v{manifest['data_version']}: {len(train_ds)} exemplos de treino "
             f"({len(train_eps)} episódios), {len(val_ds)} de validação; vocab {len(vocab)}")

    choice = select_device(tcfg["device"], tcfg["min_free_vram_mb"])
    device = choice.device
    log.info(f"dispositivo: {device} ({choice.reason})")
    model = build_model(mcfg, len(vocab), len(answers), vocab.pad_id, vocab.special_ids).to(device)
    n_params = count_parameters(model)
    log.info(f"modelo {mcfg['type']}: {n_params:,} parâmetros treináveis (init aleatória, seed {seed})")

    amp, amp_reason = (False, "AMP desligado na configuração") if tcfg["amp"] is False \
        else amp_works(model, train_ds, device)
    log.info(f"precisão mista: {amp} ({amp_reason})")

    batch, accum = int(tcfg["batch_size"]), int(tcfg["grad_accum"])
    profile = profile_memory(model, train_ds, device, batch, amp, tcfg["vram_margin_mb"],
                             tcfg["min_batch_size"], log)
    if profile["device"] != device:
        log.info(f"FALLBACK: {profile['note']}")
        device, amp = "cpu", False
        model = model.to("cpu")
    elif profile["batch_size"] != batch:
        factor = batch // profile["batch_size"]
        batch, accum = profile["batch_size"], accum * factor
        log.info(f"batch reduzido para {batch} com grad_accum {accum} (batch efetivo mantido)")
    profile.update({"final_device": device, "final_batch_size": batch, "final_grad_accum": accum,
                    "amp": amp, "amp_reason": amp_reason, "device_choice": choice.to_dict()})
    (run_dir / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    if device == "cuda":
        log.info(f"perfil VRAM: pico {profile['peak_mb']} MB com batch {batch} "
                 f"(livre {profile['free_vram_mb']} MB, margem {tcfg['vram_margin_mb']} MB)")

    max_steps = int(pilot_steps or tcfg["max_steps"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, WarmupCosine(tcfg["warmup_steps"], int(tcfg["max_steps"]), tcfg["min_lr_ratio"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    state = {"step": 0, "epoch": 0, "micro_in_epoch": 0, "best_val_acc": -1.0,
             "best_step": 0, "evals_since_best": 0, "elapsed_s": 0.0, "samples_seen": 0}

    if resume:
        ck = load_checkpoint(resume, map_location=device)
        if ck["data_version"] != manifest["data_version"]:
            raise RuntimeError(f"checkpoint é de outra versão dos dados: {ck['data_version']}")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        state.update(ck["state"])
        restore_rng_state(ck["rng"])
        log.info(f"retomando de {resume} no passo {state['step']} (época {state['epoch']})")

    collate = make_collate(vocab.pad_id)
    sampler = EpochSampler(len(train_ds), seed)
    eval_every = max_steps if pilot_steps else int(tcfg["eval_every"])
    pause_file = pause_flag(run_dir.parent)
    t_start = time.perf_counter() - state["elapsed_s"]
    stop_reason, accum_count, run_loss, run_n = None, 0, 0.0, 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def evaluate_and_save() -> dict:
        preds, val_loss = predict(model, val_ds, device, tcfg["eval_batch_size"], amp)
        m = compute_metrics(val_ds.meta, [answers.labels[p] for p in preds])
        model.train()
        state["elapsed_s"] = time.perf_counter() - t_start
        improved = m["acc"] > state["best_val_acc"]
        if improved:
            state.update(best_val_acc=m["acc"], best_step=state["step"], evals_since_best=0)
        else:
            state["evals_since_best"] += 1
        peak = torch.cuda.max_memory_allocated() / MB if device == "cuda" else None
        rec = {"type": "eval", "step": state["step"], "epoch": state["epoch"],
               "val_loss": round(val_loss, 4), "val_acc": m["acc"],
               "val_macro_task_acc": m["macro_task_acc"],
               "val_per_task": {k: v["acc"] for k, v in m["per_task"].items()},
               "elapsed_s": round(state["elapsed_s"], 1),
               "peak_vram_mb": round(peak, 1) if peak else None,
               "ram_mb": round(process_ram_mb(), 1)}
        log.metric(rec)
        log.info(f"passo {state['step']}: val_acc {m['acc']:.4f} macro {m['macro_task_acc']:.4f} "
                 f"loss {val_loss:.4f}" + (" *melhor*" if improved else ""))
        if not pilot_steps:
            payload = _ckpt_payload(model, optimizer, scheduler, scaler, cfg, state, vocab,
                                    answers, manifest, {"val_metrics": m})
            save_checkpoint(run_dir / "last.pt", payload)
            if improved:
                save_checkpoint(run_dir / "best.pt", payload)
        return m

    model.train()
    if state["step"] >= max_steps:
        stop_reason = "already_complete"  # retomar um treino concluído não roda passos extras
        log.info(f"checkpoint já está no passo {state['step']} >= {max_steps}; nada a treinar")
    else:
        log.info(f"treinando até {max_steps} passos de otimização (batch {batch} x accum {accum})")
    try:
        while stop_reason is None:
            sampler.set_epoch(state["epoch"], skip=state["micro_in_epoch"] * batch)
            loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, collate_fn=collate,
                                generator=torch.Generator())  # não consome o RNG global
            for tokens, lengths, labels, _ in loader:
                tokens, labels = tokens.to(device), labels.to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(tokens, lengths)
                loss = torch.nn.functional.cross_entropy(logits.float(), labels)
                aux = getattr(model, "aux_loss", None)  # ex.: custo de "pensar" (passos adaptativos)
                total = loss + aux if aux is not None else loss
                scaler.scale(total / accum).backward()
                state["micro_in_epoch"] += 1
                state["samples_seen"] += len(labels)
                accum_count += 1
                run_loss += loss.item() * len(labels)
                run_n += len(labels)
                if accum_count < accum:
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accum_count = 0
                state["step"] += 1
                step = state["step"]
                if step % tcfg["log_every"] == 0:
                    el = time.perf_counter() - t_start
                    log.metric({"type": "train", "step": step, "loss": round(run_loss / run_n, 4),
                                "lr": scheduler.get_last_lr()[0], "elapsed_s": round(el, 1)})
                    log.info(f"passo {step}: loss {run_loss / run_n:.4f} "
                             f"lr {scheduler.get_last_lr()[0]:.2e} ({el:.0f}s)")
                    run_loss, run_n = 0.0, 0
                minutes = (time.perf_counter() - t_start) / 60
                if step >= max_steps:
                    stop_reason = "pilot_done" if pilot_steps else "max_steps"
                elif minutes >= tcfg["max_minutes"]:
                    stop_reason = "time_budget"
                if step % eval_every == 0 or stop_reason:
                    evaluate_and_save()
                    if not stop_reason and state["evals_since_best"] >= tcfg["early_stop_patience"]:
                        stop_reason = "early_stop"
                if not stop_reason and not pilot_steps and pause_file.exists():
                    stop_reason = "paused"  # pausa pelo painel: salva o passo atual e retoma depois
                if stop_reason:
                    break
            else:
                state["epoch"] += 1
                state["micro_in_epoch"] = 0
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        state["micro_in_epoch"] -= accum_count  # micro-batches sem passo serão refeitos
        state["elapsed_s"] = time.perf_counter() - t_start
        optimizer.zero_grad(set_to_none=True)
        if not pilot_steps:
            save_checkpoint(run_dir / "last.pt", _ckpt_payload(
                model, optimizer, scheduler, scaler, cfg, state, vocab, answers, manifest))
        log.info("interrompido pelo usuário; estado salvo em last.pt")

    if stop_reason == "paused":
        state["elapsed_s"] = time.perf_counter() - t_start
        save_checkpoint(run_dir / "last.pt", _ckpt_payload(
            model, optimizer, scheduler, scaler, cfg, state, vocab, answers, manifest))
        log.info(f"pausado no passo {state['step']}; estado salvo em last.pt (retoma deste passo)")
        log.close()
        raise SystemExit(PAUSED_EXIT)  # sem summary.json: a execução segue por terminar

    elapsed = time.perf_counter() - t_start
    steps_done = state["step"]
    summary = {
        "run_dir": str(run_dir), "stop_reason": stop_reason, "steps": steps_done,
        "epochs_completed": state["epoch"], "samples_seen": state["samples_seen"],
        "elapsed_s": round(elapsed, 1),
        "sec_per_step": round(elapsed / max(1, steps_done), 4),
        "samples_per_s": round(state["samples_seen"] / max(1e-9, elapsed), 1),
        "best_val_acc": state["best_val_acc"], "best_step": state["best_step"],
        "n_params": n_params, "device": device, "amp": amp,
        "batch_size": batch, "grad_accum": accum,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / MB, 1) if device == "cuda" else None,
        "ram_mb": round(process_ram_mb(), 1),
        "data_version": manifest["data_version"],
    }
    if pilot_steps:
        full = int(tcfg["max_steps"])
        steps_per_epoch = math.ceil(len(train_ds) / (batch * accum))
        summary["estimate_full_run"] = {
            "max_steps": full,
            "estimated_minutes": round(full * summary["sec_per_step"] / 60, 1),
            "steps_per_epoch": steps_per_epoch,
            "epochs_at_max_steps": round(full / steps_per_epoch, 2),
            "time_budget_minutes": tcfg["max_minutes"],
        }
    else:
        for name in ("best.pt", "last.pt"):
            p = run_dir / name
            if p.exists():
                summary[f"{name}_mb"] = round(p.stat().st_size / MB, 2)
    (run_dir / ("pilot.json" if pilot_steps else "summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"fim ({stop_reason}): {steps_done} passos em {elapsed:.0f}s; "
             f"melhor val_acc {state['best_val_acc']:.4f} no passo {state['best_step']}")
    if pilot_steps:
        e = summary["estimate_full_run"]
        log.info(f"estimativa para {e['max_steps']} passos: ~{e['estimated_minutes']} min "
                 f"(~{e['epochs_at_max_steps']} épocas); orçamento configurado {e['time_budget_minutes']} min")
    log.close()
    return run_dir
