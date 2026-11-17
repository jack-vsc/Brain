# Curva de escala: tamanho e orçamento (parcial)

**Pergunta:** a vantagem da memória rápida sobre o Transformer some quando os
dois crescem, ou quando treinam mais?

## Montagem

- Mesmos dados (`mvp2`), pergunta depois da história, batch 64, AdamW e a
  mesma programação de lr dos baselines.
- Três tamanhos por rede (~0,5 M, ~1,5 M e ~5 M parâmetros) e dois
  orçamentos (8 mil e 24 mil passos). Configs em `configs/sc_*.toml`; o
  tamanho de 1,5 M reaproveita `b_fastweight_hops` e `b_transformer_lr5e-4`.
- 24 mil passos usam aquecimento de 1.500 passos e paciência alta (o early
  stopping não corta o treino).
- Tabela completa: `experiments/escala-tabela.md`.

## Resultados até agora

Só a seed 1, exceto 1,5 M / 8 mil passos (3 seeds). A fila
`experiments/queue_round2.txt` ainda tem as seeds 2 e 3 e as execuções de
5 M com 24 mil passos.

| rede | parâmetros | passos | test_iid | pares inéditos | histórias longas |
|---|---|---|---|---|---|
| memória rápida | 0,52 M | 8 mil | 91,3% | 77,7% | 80,0% |
| memória rápida | 0,52 M | 24 mil | 92,4% | 82,3% | 80,5% |
| memória rápida | 1,50 M | 8 mil | 89,9 ± 1,2% | 77,2 ± 1,3% | 78,2 ± 2,0% |
| memória rápida | 1,50 M | 24 mil | 92,5% | 79,7% | 80,9% |
| memória rápida | 5,18 M | 8 mil | 90,6% | 76,0% | 78,4% |
| Transformer | 0,51 M | 8 mil | 53,5% | 18,6% | 43,4% |
| Transformer | 0,51 M | 24 mil | 72,0% | 50,1% | 58,3% |
| Transformer | 1,50 M | 8 mil | 58,3 ± 2,2% | 30,2 ± 8,0% | 47,0 ± 1,9% |
| Transformer | 1,50 M | 24 mil | 79,8% | 51,5% | 60,7% |
| Transformer | 5,17 M | 8 mil | 60,2% | 23,2% | 49,4% |

## Leitura provisória

1. **A memória rápida já está saturada em 0,5 M:** a menor versão empata ou
   supera a de 1,5 M, e 5 M não acrescenta nada com 8 mil passos. O teto
   (~92%) parece vir da tarefa e do orçamento de dados, não do tamanho.
2. **O Transformer depende de orçamento, não de tamanho:** de 1,5 M para
   5 M ele ganha 2 pontos; com 3x mais passos, ganha 21. Mesmo assim, o
   melhor Transformer medido (79,8%) fica abaixo da menor memória rápida com
   o orçamento mais curto (91,3%).
3. **Pares inéditos continuam sendo a maior diferença:** 50% contra 80% no
   melhor caso de cada rede.

Com uma seed por ponto, diferenças de 1–2 pontos não são conclusivas. O
desvio entre seeds no ponto de 1,5 M é de 1–2 pontos para a memória rápida e
de 2–8 para o Transformer.

## Custo medido

Tempos desta rodada foram tomados com a GPU disputada (duas filas em paralelo
e outro processo de vídeo). Pico de VRAM no treino, seed 1: memória rápida
0,5 M 255 MB, 5 M 876 MB; Transformer 5 M 902 MB.
