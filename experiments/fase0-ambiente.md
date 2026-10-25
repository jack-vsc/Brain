# Fase 0 — ambiente e orçamento (25/10/2026)

## Medido

| item | valor |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6140 MB, compute 8.9, bf16 suportado, driver 610.88 |
| CPU | Intel Core i5-13420H, 8 núcleos físicos / 12 lógicos |
| RAM | 31,7 GB visíveis (a spec supunha 16 GB) |
| Disco | C: 3,4 GB livres; D: (SSD NVMe interno) 465 GB livres |
| Python | 3.11.9 |
| PyTorch | 2.14.1+cu130 (CUDA 13.0, cuDNN 9.24), numpy 2.4.6, psutil 7.2.2, pytest 9.1.1 |

## Decisões

- **Ambiente em D:**, porque o PyTorch CUDA não cabia com folga no C:.
  A venv fica em `D:\Brain-env\.venv` (junction `.venv` no projeto) e o
  cache/temp do uv em `D:\Brain-env`. Nada foi instalado globalmente e
  nenhum driver foi alterado.
- **GPU compartilhada.** Outro processo na mesma máquina (de visão computacional)
  ocupou até ~5,5 GB de VRAM com 100% de uso durante as medições. O
  treino detecta a VRAM livre, faz perfil de memória e cai para CPU de
  forma explícita se não couber.

## Custos medidos (GRU 1,5 M parâmetros, batch 64, AMP fp16)

| condição | s/passo | 12k passos |
|---|---|---|
| GPU livre | 0,045–0,071 | ~9–14 min |
| GPU disputada (outro processo a 100%) | 0,25–0,35 | ~50–70 min |
| CPU (8 threads) | 3,18 | ~10,6 h |

Pico de VRAM do treino: 226 MB (batch 64, sequências de até ~180 tokens).
Checkpoint com otimizador: 17,2 MB.

## Incidente

Durante o primeiro treino longo, com a GPU disputada, houve um
`CUDA error: an illegal memory access` logo após o checkpoint do passo 4000.
Ele coincidiu com o outro processo liberando a GPU, e o treino seguinte com
o mesmo código rodou 12 mil passos sem erro. O checkpoint atômico preservou o
estado. Mitigação: `scripts/run_queue.py` roda cada job num processo novo e
retoma de `last.pt` em caso de queda.
