# Arquitetura do código e dos modelos

Visão de como as peças se encaixam, do gerador de dados ao painel. As
decisões de dados e treino estão em [DESIGN.md](DESIGN.md); os resultados de
cada ideia, em [ITERACOES.md](ITERACOES.md).

## Fluxo geral

```
configs/data_*.toml ──► brain generate-data ──► data/generated/<nome>/
                                                 train/val/test_*.jsonl + manifest.json
                                                          │
configs/<modelo>.toml ──► brain train ──► runs/<nome>-s<seed>/
                                          config.json, metrics.jsonl, train.log,
                                          last.pt, best.pt, summary.json
                                                          │
                          brain evaluate ──► eval-best.{json,md}
                                                          │
       scripts/compare_runs.py ──► experiments/*-tabela.md
       python -m brain.dashboard ──► painel (servidor local ou HTML estático)
```

`scripts/run_queue.py` encadeia treinos e avaliações a partir de um arquivo de
fila (`experiments/queue_*.txt`), um processo por tarefa.

## Pacotes

| pacote | conteúdo |
|---|---|
| `brain.data.world` | léxico, eventos e a semântica de estado usada para rotular perguntas |
| `brain.data.generator` | gerador procedural versionado, splits sem vazamento, pares reservados |
| `brain.data.tokenizer` | tokenizer determinístico; vocabulário só do treino |
| `brain.data.dataset` | uma pergunta = um exemplo de classificação; `collate` com padding |
| `brain.symbolic.solver` | parser + semântica reimplementada a partir do texto (referência, não alimenta a rede) |
| `brain.models` | GRU, Transformer, memória rápida, episódica, Mamba-2, minGRU e o `registry` |
| `brain.training.trainer` | laço de treino, perfil de VRAM, AMP, retomada exata, pausa |
| `brain.training.continual` | aprendizado sequencial por blocos e as estratégias de replay/consolidação |
| `brain.training.checkpoint` | escrita atômica, carga com `weights_only=True`, estado dos RNGs |
| `brain.training.device` | inspeção de hardware e escolha explícita de dispositivo |
| `brain.training.control` | arquivo-sinal de pausa usado pelo treino e pela fila |
| `brain.evaluation` | métricas por tarefa e tag, avaliação neural × simbólica × majoritária, latência, benchmark de memória |
| `brain.memory.store` | memória persistente em SQLite (log só de acréscimo + índice derivado) |
| `brain.dashboard` | coleta do estado em disco, traços de atividade e a página do painel |
| `brain.cli`, `brain.chat` | linha de comando e chat interativo no micromundo |

## Entrada da rede

```
<bos> Projeto alfa: Ana deixou a chave na oficina. <sep> ... <sep> <q> onde está a chave agora ? <eos>
```

A saída é uma classificação entre 30 rótulos: pessoas, lugares, `sim`,
`não`, `ninguém` e `UNKNOWN`. Com `question_position = "before"`, a pergunta
vem antes da história.

## Modelos

Todos começam com pesos aleatórios e têm ~1,5 M parâmetros nas comparações
principais.

### GRU (`gru`)

`embedding(128) → GRU(2 × 384) → LayerNorm → Linear`. O estado final é o do
último token real (`pack_padded_sequence`). É o baseline sem memória externa:
tudo o que a rede sabe da história precisa caber no vetor de estado.

### Transformer (`transformer`)

Encoder pre-LN, atenção bidirecional, posições rotacionais (RoPE), `d_model`
192, 4 camadas, 4 cabeças, FFN 576. Classifica pelo token `<eos>`. Máscara
de padding testada: a saída não depende do resto do batch.

### Memória rápida com regra delta (`fastweight_gru`)

O modelo principal do projeto.

1. Uma GRU de 2 camadas lê a história.
2. A cada token, ela produz, para cada uma de 4 cabeças de memória, uma chave
   `k`, um valor `v` (48 dimensões) e uma força de escrita `β ∈ (0, 1)`.
3. A memória `W` (uma matriz por cabeça, zerada no começo do episódio) é
   reescrita pela regra delta:

   ```
   antigo = Wᵀk
   W ← W + β · k (v − antigo)ᵀ
   ```

   Escrever de novo a mesma chave substitui o valor antigo em vez de somar,
   o que é o comportamento certo para "a chave agora está no parque".
4. No fim, a leitura é feita em `read_hops` passos: cada leitura `Wᵀq`
   atualiza um estado de controle, que gera a próxima consulta. Com 3
   passos, a rede resolve cadeias como objeto → portador → local do portador.
5. A cabeça de saída usa o estado da GRU e as leituras.

**Forma em blocos.** O laço token a token é sequencial. Dentro de um bloco de
C tokens, os pseudo-valores `u_t = β_t (v_t − Sᵀk_t − Σ_{i<t} (k_iᵀk_t) u_i)`
satisfazem um sistema triangular unitário,
`(I + diag(β)·tril(KKᵀ, −1)) U = diag(β)(V − KS)`, resolvido de uma vez na GPU.
É a mesma conta (testado contra o laço), cerca de 4x mais rápida.

**Opções experimentais** (todas com resultado registrado no diário):
`key_dim`/`key_topk` (chaves esparsas), `familiarity`, `adaptive_hops`
(leituras adaptativas no estilo PonderNet), `forget_gate`/`forget_bias`
(porta de esquecimento, Gated DeltaNet), `time_code` (código de tempo somado
à chave e ao valor) e `rnn = "mingru"`.

### Memória episódica (`episodic_gru`)

Variante em que a memória é lida e escrita nas fronteiras de evento (`<sep>`)
durante a leitura, com reinstauração do conteúdo recuperado no estado
recorrente. A sequência é cortada em segmentos; cada segmento passa pela GRU
do cuDNN em paralelo e o laço em Python é só sobre segmentos.

### Mamba-2 (`mamba2`)

Blocos no estilo Mamba-2 (SSD) escritos do zero: projeção de entrada,
convolução causal curta, estado por cabeça
`S_t = exp(Δ_t A) S_{t−1} + Δ_t B_t x_tᵀ` e saída `y_t = C_tᵀ S_t + D x_t`,
com varredura em blocos exata. Não usa o código oficial (que precisa de
kernels CUDA sem suporte no Windows).

### minGRU (`rnn = "mingru"` na memória rápida)

Recorrência cuja porta depende só da entrada, resolvida para todos os tokens
de uma vez em espaço log com `logcumsumexp` (`brain/models/mingru.py`). Entra
no lugar da GRU dentro da memória rápida.

## Treino

- Cross-entropy + AdamW, aquecimento linear e cosseno até 5% do lr, clipping
  em 1,0.
- Antes de treinar, um passo com o batch de sequências mais longas mede o
  pico de VRAM. Se não couber, o batch cai pela metade e o gradient
  accumulation compensa. CPU só como fallback explícito, registrado em
  `profile.json`.
- AMP fp16 só se um passo de teste passar.
- `last.pt` e `best.pt` a cada avaliação (500 passos), com otimizador,
  scheduler, scaler, RNGs e a posição dentro da época. A retomada reproduz a
  mesma ordem de exemplos.
- Parada por `max_steps`, orçamento de tempo, early stopping, Ctrl+C ou
  pausa. Nos três últimos casos o estado é salvo antes de sair.

### Pausa

`runs/_control/pause` é um arquivo-sinal. Com ele presente:

- o treino em andamento salva `last.pt` no passo atual e sai com código 75,
  sem `summary.json`;
- `scripts/run_queue.py` espera o sinal sumir e retoma a mesma execução
  daquele passo, sem contar como falha;
- avaliações e benchmarks contínuos em curso terminam antes; a fila pausa
  antes da tarefa seguinte.

```bash
python -m brain.training.control pause    # ou resume / status
```

## Aprendizado contínuo

`brain.training.continual` treina a mesma rede em três blocos
(`localizacao → posse → correcoes`) e mede a matriz `R[i][j]` (acerto no
bloco j depois do bloco i). Estratégias: `none`, `uniform`, `category`,
`ewc`, `episodic`, `episodic_ewc`, `episodic_diverse` e `uniform_sleep`. As
episódicas gravam as histórias escolhidas em SQLite (`brain.memory.store`),
releem numa conexão nova e recalculam o alvo com o solucionador simbólico a
partir das frases guardadas.

## Memória persistente

`brain.memory.store.MemoryStore` guarda cada frase como evento num log só de
acréscimo. Correções são eventos novos que apontam para o corrigido;
negações e hipóteses ficam no log mas não viram fatos. Há expurgo explícito
por entidade, que registra só que houve expurgo. A recuperação, por projeto e
entidade, devolve as frases originais como evidência para a rede, nunca uma
resposta pronta. `brain memory-benchmark` compara quatro condições: só os
pesos, história no contexto, SQLite depois de um reinício e a memória rápida
`W` salva em disco.

## Painel

- `collect.py` lê o que existe em disco: execuções, filas, diário, roadmap,
  dados, resultados da Fase 4 e GPU (`nvidia-smi`).
- `traces.py` roda dois checkpoints de referência na CPU em algumas histórias
  e grava a atividade por camada, a atenção do Transformer e, na memória
  rápida, a força de escrita e a contribuição exata de cada token em cada
  leitura. As reconstruções são conferidas contra os logits do modelo.
- `page.html` é uma página única (HTML, CSS e JavaScript sem build) com o mapa
  animado, as iterações, a fila com o detalhe de cada treino, os resultados e
  os dados.
- `python -m brain.dashboard serve` serve a página e `/api/state`, com
  atualização a cada 20 s e os botões de atualizar e pausar.
  `python -m brain.dashboard build` grava um HTML estático.

## Testes

`tests/` cobre a semântica do micromundo e a concordância com o solucionador,
splits sem vazamento, determinismo do gerador, independência do padding em
todos os modelos, equivalência das formas em blocos com os laços de
referência, checkpoint atômico e recusa de pickle malicioso, retomada e
pausa exatas, memória SQLite, benchmark contínuo em miniatura e o coletor do
painel. Rodam em CPU.
