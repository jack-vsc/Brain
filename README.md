# Brain

Redes neurais pequenas, treinadas do zero, que aprendem a acompanhar o estado
de um micromundo em português: quem está com qual objeto, onde cada coisa
está, o que mudou depois de uma correção, o que aconteceu antes. O objetivo é
comparar mecanismos de memória inspirados no cérebro com um Transformer
moderno, com o mesmo orçamento de parâmetros e de treino.

Nenhum modelo, tokenizer ou embedding pré-treinado é usado. Os pesos começam
aleatórios, o vocabulário vem só do split de treino e os dados são gerados
por um gerador procedural com um solucionador simbólico independente para
conferir os rótulos.

```
Projeto alfa: Ana deixou a chave na oficina.
Projeto alfa: Bruno pegou a chave.
Projeto alfa: Bruno foi para a escola.
Projeto alfa: Correção: Ana deixou a chave no parque.
Pergunta: onde está a chave agora?          → escola
Pergunta: onde a chave estava primeiro?     → parque
```

## Resultado principal

Mesmos dados, ~1,5 M parâmetros, 8 mil passos com batch 64, média de 3 seeds:

| modelo | histórias novas | pares objeto–lugar inéditos | histórias longas (10–16 eventos) |
|---|---|---|---|
| GRU | 62,5 ± 0,8% | 11,3% | 51,5% |
| Transformer (pre-LN, RoPE) | 58,3 ± 2,2% | 30,2 ± 8,0% | 47,0 ± 1,9% |
| Mamba-2 (reimplementação) | 66,1 ± 1,8% | 16,6 ± 5,3% | 54,6 ± 1,2% |
| **GRU + memória rápida (regra delta) + 3 leituras** | **89,9 ± 1,2%** | **77,2 ± 1,3%** | **78,2 ± 2,0%** |

- Com 3x mais treino, o Transformer chega a 79,8%; a memória rápida, a 92,5%.
- Uma memória rápida de 0,5 M parâmetros (91,3%) supera um Transformer de
  5 M (60,2%) com o mesmo número de passos (`experiments/escala.md`, parcial).
- Em aprendizado contínuo (três blocos de tarefas em sequência), replay
  uniforme leva o acerto final de 58% para 84% e o esquecimento de 40 para
  2 pontos. As versões "hipocampais" (memória episódica em SQLite, seleção
  por surpresa ou diversidade, sono) empatam com o replay uniforme, mas não
  o superam (`experiments/fase4-continuo.md`).

As ideias testadas depois da memória rápida com 3 leituras (chaves
esparsas, memória episódica, reinstauração, leituras adaptativas, porta de
esquecimento, código de tempo, minGRU) estão no
[diário de iterações](docs/ITERACOES.md), com hipótese, mecanismo e resultado
de cada uma, inclusive as que falharam.

### Limites

É um micromundo de gramática fixa, com 12 pessoas, 12 objetos e 12 lugares, e
a saída é uma classificação entre 30 rótulos. Os resultados dizem respeito a
este ambiente controlado; não mostram que o mecanismo escala para linguagem
aberta. "Inspirado no cérebro" descreve a ideia por trás de cada mecanismo,
não uma afirmação biológica.

## Instalação

Requer Python 3.11. Com GPU NVIDIA, instale o PyTorch do índice CUDA
correspondente ao seu driver; sem GPU, tudo roda na CPU (bem mais devagar).

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev]"
```

As versões exatas usadas nos experimentos estão em
[requirements.lock.txt](requirements.lock.txt) (Windows 11, RTX 4050 6 GB,
torch 2.14.1+cu130).

## Uso

```bash
brain hardware                                       # GPU, VRAM, versões
brain generate-data --config configs/data_mvp2.toml  # gera data/generated/mvp2
brain train --config configs/b_fastweight_hops.toml --seed 1
brain evaluate --checkpoint runs/b_fastweight_hops-s1/best.pt
brain chat --checkpoint runs/b_fastweight_hops-s1/best.pt
```

Teste rápido de ponta a ponta (menos de um minuto):

```bash
brain generate-data --config configs/data_smoke.toml
brain train --config configs/smoke.toml
```

Opções úteis de `brain train`:

- `--pilot-steps N` roda N passos e grava em `pilot.json` uma estimativa do
  custo do treino completo;
- `--resume runs/<execução>/last.pt` retoma do passo exato;
- `--set train.lr=0.001` sobrescreve qualquer chave da configuração.

Outros comandos:

```bash
brain continual-benchmark --config configs/continual.toml --strategy uniform --seed 1
brain memory-benchmark --checkpoint runs/b_fastweight_hops-s1/best.pt
brain export-graph --store memoria.sqlite --output grafo.json
```

Cada execução grava em `runs/<nome>-s<seed>/` a configuração, o hardware, o
perfil de VRAM, `metrics.jsonl`, `train.log`, `summary.json`, `last.pt`,
`best.pt` e, depois de avaliar, `eval-best.{json,md}`.

### Filas de experimentos

```bash
python scripts/run_queue.py experiments/queue_round2.txt
python scripts/compare_runs.py b_transformer_lr5e-4 b_fastweight_hops --out tabela.md
```

Cada linha da fila é um treino, avaliação, piloto ou benchmark contínuo, num
processo próprio. Se um treino cair, a fila retoma de `last.pt`.
`python -m brain.training.control pause` pede uma pausa: o treino salva o
passo atual e a fila espera até `resume`.

### Painel

```bash
python -m brain.dashboard serve      # http://127.0.0.1:8765, atualiza a cada 20 s
python -m brain.dashboard build      # HTML estático em dashboard/painel.html
```

Mostra a atividade gravada dos dois modelos de referência lado a lado (a
memória rápida sendo escrita e lida token a token, e a atenção do
Transformer), o resultado de cada iteração, a fila com o progresso de cada
treino, os resultados, os custos de memória e a curva de escala. Com
`--host 0.0.0.0`, abre em outros aparelhos da mesma rede.

## Testes

```bash
pytest
```

Cobrem a semântica do micromundo e 100% de concordância com o solucionador
simbólico, splits sem vazamento, determinismo, independência do padding,
equivalência das formas paralelas com os laços de referência, checkpoints
atômicos e seguros, retomada e pausa exatas, a memória SQLite, o benchmark
contínuo em miniatura e o coletor do painel. Rodam em CPU.

## Estrutura

```
src/brain/
  data/          micromundo, gerador, tokenizer, dataset
  symbolic/      solucionador simbólico de referência
  models/        GRU, Transformer, memória rápida, episódica, Mamba-2, minGRU
  training/      treino, aprendizado contínuo, checkpoints, dispositivo, pausa
  evaluation/    métricas, avaliação, benchmark de memória
  memory/        memória persistente em SQLite
  dashboard/     painel
  cli.py, chat.py
configs/         configurações de dados, modelos e experimentos
experiments/     relatórios, tabelas e filas
docs/            design, arquitetura, diário de iterações, roadmap
scripts/         fila de experimentos e tabelas comparativas
tests/
```

`data/`, `runs/` e `dashboard/` são gerados e ficam fora do Git.

## Documentação

- [docs/DESIGN.md](docs/DESIGN.md): micromundo, perguntas, splits, solucionador, treino.
- [docs/ARQUITETURA.md](docs/ARQUITETURA.md): código, modelos e painel.
- [docs/ITERACOES.md](docs/ITERACOES.md): diário de iterações de arquitetura.
- [docs/ROADMAP.md](docs/ROADMAP.md): fases e pendências.
- [experiments/](experiments/): relatórios de cada fase e tabelas completas.

## Referências

Os mecanismos foram implementados do zero a partir dos artigos:

- Schmidhuber (1992), *Learning to control fast-weight memories*; Ba et al.
  (2016), *Using fast weights to attend to the recent past*.
- Yang et al. (2024), *Parallelizing linear transformers with the delta rule
  over sequence length* (DeltaNet) e *Gated Delta Networks* (2024).
- Dao & Gu (2024), *Transformers are SSMs* (Mamba-2).
- Feng et al. (2024), *Were RNNs all we needed?* (minGRU).
- Banino et al. (2021), *PonderNet*.
- Su et al. (2021), *RoFormer* (RoPE).
- Kirkpatrick et al. (2017), *Overcoming catastrophic forgetting in neural
  networks* (EWC).

## Licença

[MIT](LICENSE). As dependências (PyTorch, psutil, pytest) têm licenças
próprias, compatíveis (BSD e MIT). O painel carrega as fontes Bricolage
Grotesque, Instrument Sans e JetBrains Mono do Google Fonts (SIL Open Font
License); nenhuma fonte ou biblioteca de terceiros é distribuída no
repositório.
