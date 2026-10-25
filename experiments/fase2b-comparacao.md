# Fase 2b — memória rápida vs Transformer vs GRU (25/10/2026)

Pergunta: num micromundo controlado, com o mesmo orçamento, mecanismos de
memória inspirados no cérebro superam um Transformer moderno?

## Protocolo

- Dados `data/generated/mvp2` (v4d25bf904f4e): 80 mil perguntas de treino.
  Avaliação em `val`, `test_iid`, `test_comp` (pares objeto–lugar nunca
  vistos no treino) e `test_long` (histórias de 18 frases em média, contra
  7,7 no treino).
- Todos os modelos com ~1,5 M parâmetros, pesos aleatórios, a pergunta
  **depois** da história, batch 64, 8.000 passos, AdamW, mesma programação
  de lr, AMP fp16, RTX 4050.
- lr escolhido **só pela validação** (seed 1): GRU {1e-3, 2e-3} → 1e-3;
  Transformer {3e-4, 5e-4, 1e-3, 2e-3} → 5e-4; memória rápida {1e-3, 2e-3}
  → 1e-3; iteração 2 e 3 com 2e-3.
- 3 seeds por configuração (média ± desvio-padrão). Orçamento 3x (24 mil
  passos) com 1 seed, para o Transformer e a iteração 2.
- Tabelas completas, inclusive por tarefa: `experiments/fase2b-tabela.md`.
  Cada execução: `runs/<nome>-s<seed>/eval-best.md`.

## Orçamento de treino (média das seeds)

| modelo | parâmetros | passos | melhor passo | exemplos vistos | épocas | tempo de treino | s/passo | VRAM (pico) | latência p50 (batch 1) |
|---|---|---|---|---|---|---|---|---|---|
| GRU | 1,50 M | 8.000 | 7.833 | 512 mil | 6,4 | 300 s | 0,037 | 182 MB | 8,0 ms |
| Transformer (RoPE) | 1,50 M | 8.000 | 8.000 | 512 mil | 6,4 | 339 s | 0,042 | 432 MB | 4,4 ms |
| It. 1: memória rápida | 1,48 M | 8.000 | 7.333 | 512 mil | 6,4 | 487 s | 0,061 | 354 MB | 10,2 ms |
| It. 2: + 3 passos de leitura | 1,50 M | 8.000 | 7.500 | 512 mil | 6,4 | 495 s | 0,062 | 352 MB | 9,4 ms |
| It. 3: chaves esparsas | 1,46 M | 8.000 | 7.500 | 512 mil | 6,4 | 466 s | 0,058 | 488 MB | 9,5 ms |
| Transformer ×3 (1 seed) | 1,50 M | 24.000 | 24.000 | 1,54 milhão | 19,2 | 764 s | 0,032 | 432 MB | 4,2 ms |
| It. 2 ×3 (1 seed) | 1,50 M | 24.000 | 20.000 | 1,54 milhão | 19,2 | 1.227 s | 0,051 | 352 MB | 7,4 ms |

Passos = atualizações dos pesos (batch 64). "Melhor passo" = checkpoint
avaliado (maior acurácia na validação). Passos, exemplos e parâmetros são
exatos. Tempos são aproximados: em algumas execuções a GPU foi disputada
por outros processos.

## Resultado principal (acurácia, 3 seeds)

| modelo | parâmetros | val | test_iid | test_comp | par inédito | test_long | isolamento (test_iid) | UNKNOWN inventado (test_iid) ↓ |
|---|---|---|---|---|---|---|---|---|
| GRU | 1,50 M | 0,630 ± 0,004 | 0,625 ± 0,008 | 0,482 ± 0,001 | 0,113 ± 0,002 | 0,515 ± 0,004 | 0,257 ± 0,032 | 0,331 ± 0,035 |
| Transformer (pre-LN, RoPE) | 1,50 M | 0,592 ± 0,021 | 0,583 ± 0,022 | 0,508 ± 0,016 | 0,302 ± 0,080 | 0,470 ± 0,019 | 0,172 ± 0,021 | 0,503 ± 0,055 |
| It. 1: GRU + memória rápida (regra delta) | 1,48 M | 0,865 ± 0,011 | 0,865 ± 0,011 | 0,795 ± 0,014 | 0,610 ± 0,009 | 0,751 ± 0,011 | 0,680 ± 0,221 | 0,132 ± 0,031 |
| **It. 2: + 3 passos internos de leitura** | 1,50 M | **0,900 ± 0,014** | **0,899 ± 0,012** | **0,868 ± 0,014** | **0,772 ± 0,013** | **0,782 ± 0,020** | **0,878 ± 0,050** | **0,082 ± 0,014** |
| It. 3: + chaves esparsas + familiaridade | 1,46 M | 0,751 ± 0,029 | 0,748 ± 0,031 | 0,625 ± 0,042 | 0,296 ± 0,083 | 0,632 ± 0,025 | 0,207 ± 0,032 | 0,301 ± 0,068 |
| classe majoritária | — | 0,270 | 0,265 | 0,191 | — | 0,184 | — | — |
| solucionador simbólico (referência, não neural) | — | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,000 |

## Orçamento 3x (24 mil passos, 1 seed)

| modelo | val | test_comp | par inédito | test_long | isolamento (test_iid) | tempo de treino |
|---|---|---|---|---|---|---|
| Transformer ×3 | 0,796 | 0,724 | 0,515 | 0,607 | 0,496 | 764 s |
| It. 2 ×3 | 0,926 | 0,892 | 0,797 | 0,809 | 0,921 | 1.227 s |
| It. 2 ×1 (referência) | 0,900 | 0,868 | 0,772 | 0,782 | 0,878 | 495 s |

## O que os dados mostram

1. **A memória rápida com passos de leitura supera o Transformer em todos
   os splits e em todas as métricas, com o mesmo orçamento.** A diferença é
   grande e bem acima da variação entre seeds: +31 pontos na validação, +36
   em test_comp, +47 em pares inéditos, +31 em histórias longas e +71 no
   isolamento entre projetos.
2. **O Transformer estava limitado pelo orçamento, como suspeitávamos.** Com
   3x mais passos ele sobe de 59% para 80%. Mesmo assim fica abaixo da
   iteração 2 com 1x (90%), que também treina em menos tempo (495 s contra
   764 s). Com 3x para os dois, a diferença cai de 31 para 13 pontos na
   validação, mas continua em 20 pontos nas histórias longas e em 42 pontos
   no isolamento. **Eficiência de amostras** é a afirmação mais sólida aqui.
3. **Os passos internos de leitura (iteração 2) ajudam de forma consistente**
   sobre a iteração 1: +3,5 pontos em média, +16 em pares inéditos, +20 no
   isolamento, e menos da metade de respostas inventadas. Também estabilizam:
   o desvio do isolamento cai de 0,22 para 0,05 entre seeds.
4. **A separação de padrões com top-k duro (iteração 3) falhou nas 3
   seeds**, por falha de otimização (veja `docs/ITERACOES.md`).
5. **Ponto fraco que sobra para todos:** "onde estava antes?" (75% na
   iteração 2, 53% em histórias longas) e o isolamento em histórias longas
   (68%).

## O que os dados NÃO mostram

- **Não mostram que a arquitetura é melhor que Transformers em geral.** É um
  micromundo sintético, com vocabulário de 72 palavras, modelos de 1,5 M
  parâmetros e uma tarefa que favorece explicitamente o rastreamento de
  estado, justamente o que a memória rápida faz.
- O Transformer foi ajustado só em lr e orçamento. Profundidade, largura,
  atenção causal, mais dados, regularização e outras técnicas não foram
  exploradas, e um Transformer mais bem ajustado pode reduzir a diferença.
- O orçamento 3x tem uma seed só.
- A ideia de memória rápida com regra delta **não é nova** (Schmidhuber 1992;
  Ba et al. 2016; DeltaNet; Titans). O que este projeto testou é uma
  combinação específica (GRU + regra delta + passos de leitura) numa tarefa
  controlada.
- Tempos e latências variam com a disputa da GPU (outros processos
  rodaram em paralelo em algumas execuções). Em inferência com batch 1, o
  Transformer é mais rápido (p50 ~4,4 ms contra ~9,4 ms).

## Bugs encontrados e corrigidos nesta fase

- `nn.GRU` com dropout entre camadas no cuDNN derrubava o processo ao sair
  (0xC0000409, torch 2.14.1+cu130 no Windows). Reproduzido com um script
  mínimo; substituído por `StackedGRU`, com teste de regressão.
- Retomar um treino já concluído rodava 1 passo extra; corrigido e testado.
  As execuções afetadas foram refeitas e as antigas arquivadas em
  `runs/_superseded/`.
- A fila de experimentos passou a distinguir "o treino caiu" de "o treino
  terminou e o processo caiu na saída".

## Próximos experimentos

- Iteração 5 (na fila): iteração 2 + reinstauração no fim de cada frase,
  com ablação limpa e 3 seeds.
- Separação de padrões com esparsidade suave.
- Um Transformer mais forte (mais camadas, atenção causal, mais passos) com
  3 seeds, para tornar a comparação mais exigente.
- Fase 3: persistência entre sessões (H1) com o melhor modelo.
