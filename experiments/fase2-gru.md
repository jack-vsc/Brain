# Fase 2 — baseline GRU sem memória externa (25/10/2026)

Dados: `data/generated/mvp` (v328fc97602d9), com os mesmos arquivos de
`mvp2`. Avaliação final em `mvp2`, que acrescenta `test_long`. Modelo:
embedding 128 → GRU 2×384 → cabeça, 1.499.932 parâmetros, pesos aleatórios,
seed 1, AdamW lr 2e-3, batch 64, 12.000 passos (~9,6 épocas), AMP fp16 na
RTX 4050.

Única diferença entre as variantes: onde a pergunta aparece na entrada.

| variante | val | test_iid | test_comp | par inédito | test_long | UNKNOWN inventado (val) |
|---|---|---|---|---|---|---|
| pergunta **depois** da história | 0,553 | 0,549 | 0,378 | **0,000** | 0,433 | 0,520 |
| pergunta **antes** da história | **0,830** | **0,825** | **0,791** | **0,673** | **0,730** | **0,211** |
| classe majoritária (UNKNOWN) | 0,270 | 0,265 | 0,191 | — | 0,184 | — |
| solucionador simbólico (referência) | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,000 |

Execuções: `runs/gru_qafter-s1`, `runs/gru_qbefore-s1` (`eval-best.md` de cada uma).

## Leitura

- **H3 (estado recorrente) — parcialmente apoiada, com uma ressalva
  importante.** A GRU acompanha mudanças de local e posse bem acima do acaso,
  mas só quando sabe de antemão o que acompanhar. Com a pergunta no fim, ela
  precisaria guardar o estado de todas as entidades num único vetor, e não
  consegue (55%).
- **Generalização composicional depende da mesma coisa.** Com a pergunta
  depois, a GRU nunca acerta um par (objeto, lugar) inédito (0/2.345): ela
  aprendeu associações, não a regra. Com a pergunta antes, acerta 67%.
- **Comprimento.** As duas variantes perdem desempenho em histórias 2,3x mais
  longas (−12 e −10 pontos), mas não colapsam.
- **Isolamento entre projetos é ruim nas duas** (20–33% nas perguntas sobre
  uma entidade que só existe no outro projeto). As redes "vazam" fatos.
- **UNKNOWN:** com a pergunta depois, a rede inventa resposta em 52% dos casos
  em que o certo seria UNKNOWN.

Este resultado motiva diretamente a Fase 2b: dar à rede uma memória onde ela
possa **escrever** o que lê sem saber ainda a pergunta.

## Limitações e incidentes

- Uma seed só nesta fase (as 3 seeds ficaram para os modelos da Fase 2b).
- O treino "pergunta depois" caiu no passo 4000 por um erro de CUDA e foi
  retomado de `last.pt`. Depois de concluir, a fila de experimentos o
  "retomou" 3 vezes por engano (bug do cuDNN que derrubava o processo na
  saída, mais um bug de retomada que rodava 1 passo extra). O `best.pt` é do
  passo 9000 e não foi afetado. Os dois bugs foram corrigidos e cobertos por
  testes (veja `fase2b-comparacao.md`).
- Latência por pergunta (batch 1, GPU): p50 ~3–5 ms. Pico de VRAM na
  avaliação: ~225 MB.
