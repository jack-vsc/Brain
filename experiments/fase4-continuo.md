# Fase 4: aprendizado contínuo (H2)

**Pergunta:** quando a mesma rede aprende tarefas em sequência, sem rever os
dados antigos, quanto ela esquece? E quais estratégias de memória evitam isso
com o mesmo orçamento?

## Montagem

- **Rede:** a mesma arquitetura da iteração 2 (`fastweight_gru`, 3 leituras,
  ~1,50 M parâmetros), com pesos aleatórios no início.
- **Blocos, em ordem:**
  1. `localizacao`: onde está o objeto ou a pessoa, e perguntas sim/não;
  2. `posse`: só "quem está com…?";
  3. `correcoes`: metade das histórias tem correções; perguntas de local,
     "primeiro" e "antes".
  Dados em `data/generated/continual`: 6.000 histórias de treino e 500 de
  teste por bloco, sem histórias repetidas entre blocos e splits. O
  vocabulário é a união dos treinos dos blocos (divergência documentada: sem
  isso, palavras do bloco 3 seriam desconhecidas nos blocos 1 e 2).
- **Orçamento igual para todas as estratégias:** 3.000 passos por bloco,
  batch 64, lr 2e-3 com aquecimento e cosseno reiniciados a cada bloco. Com
  replay, 16 dos 64 exemplos do batch vêm da memória a partir do 2º bloco, e
  a memória guarda até 2.000 exemplos.
- **Estratégias:**
  - `none`: treino sequencial puro;
  - `uniform`: buffer por amostragem de reservatório sobre tudo o que já foi
    visto;
  - `category`: buffer com cotas iguais por tipo de pergunta;
  - `ewc`: consolidação sináptica (penalidade λ Σ F (θ − θ*)², λ = 1000,
    Fisher diagonal com 20 batches);
  - `episodic`: inspirada no hipocampo.
    - No fim de cada bloco, metade da cota guarda as experiências de maior
      perda ("surpresa") e a outra metade uma amostra aleatória.
    - Tudo vai para uma memória episódica SQLite (`brain.memory.store`), com
      cotas iguais por bloco.
    - As histórias são relidas da memória numa conexão nova, e o alvo é
      recalculado pelo solucionador simbólico sobre as frases guardadas.
    - Nas 6 execuções, os 5.998 alvos recalculados bateram 100% com as
      respostas originais;
  - `episodic_ewc`: episodic + ewc.
- **Métricas:**
  - matriz R, com o acerto em cada bloco depois de treinar cada bloco;
  - acerto final médio, que é a média da última linha de R;
  - esquecimento de um bloco: o melhor acerto que ele já teve antes do fim
    menos o acerto final;
  - plasticidade: o acerto num bloco logo depois de treiná-lo.
- **Seeds:** 3 por estratégia. As 18 execuções da primeira rodada e as 12 da
  segunda terminaram sem falhas.

## Resultados (média ± desvio, 3 seeds)

| estratégia | acerto final médio | esquecimento médio ↓ | esq. localização | esq. posse | tempo (min) |
|---|---|---|---|---|---|
| sem replay | 58,4 ± 0,5% | 40,0 ± 3,2 | 21,4 ± 6,4 | 58,6 ± 0,8 | 8,8 |
| consolidação (EWC) | 60,7 ± 2,5% | 35,8 ± 3,8 | 10,8 ± 7,3 | 60,8 ± 1,5 | 9,7 |
| **replay uniforme** | **84,2 ± 2,6%** | **2,0 ± 0,9** | 0,8 ± 2,0 | 3,3 ± 0,9 | 8,7 |
| replay por tipo de pergunta | 83,4 ± 1,8% | 0,3 ± 5,8 | −7,1 ± 10,5 | 7,7 ± 2,7 | 8,7 |
| episódica SQLite + surpresa | 81,3 ± 4,3% | 6,6 ± 2,1 | 10,3 ± 3,3 | 2,9 ± 1,4 | 15,1 |
| episódica + EWC | 80,7 ± 4,9% | 5,6 ± 2,0 | 5,0 ± 1,7 | 6,2 ± 2,3 | 16,1 |
| episódica SQLite + diversidade (2ª rodada) | 83,3 ± 2,9% | 3,7 ± 2,0 | 4,4 ± 4,2 | 3,1 ± 0,5 | 21,0* |
| replay uniforme + sono (2ª rodada) | 84,0 ± 2,7% | 1,3 ± 2,8 | −0,9 ± 5,8 | 3,5 ± 0,2 | 14,1* |

\* Rodaram com outra fila em paralelo na GPU: o tempo está inflado e não
serve para comparar com as linhas de cima.

A plasticidade foi parecida em todas as estratégias: cerca de 68–74% em
localização, 98–99% em posse e 82–85% em correções. Nenhuma estratégia
travou o aprendizado do bloco novo.

Matriz R média (linha = depois de treinar; coluna = bloco testado):

| | localização | posse | correções |
|---|---|---|---|
| sem replay, após localização | 72,4 | 33,0 | 61,3 |
| sem replay, após posse | 20,2 | 99,3 | 35,6 |
| sem replay, após correções | 51,1 | 40,7 | 83,4 |
| replay uniforme, após correções | 72,4 | 95,6 | 84,5 |
| episódica, após correções | 63,9 | 96,0 | 84,0 |

Esquecimento negativo significa que o bloco melhorou depois, porque os
blocos compartilham habilidades. Treinar `correcoes` recupera parte de
`localizacao` mesmo sem replay (20 → 51%).

## Segunda rodada: diversidade, sono e o Transformer no mesmo teste

- **Episódica com diversidade** (`episodic_diverse`): a mesma memória SQLite,
  mas a seleção percorre os tipos de pergunta e de marcação em rodízio, em
  vez de priorizar a surpresa. Subiu de 81,3% para 83,3% e esqueceu menos
  localização (10,3 → 4,4 pontos). Confirma o diagnóstico da primeira rodada:
  o problema da versão por surpresa era a cobertura, não a memória em si.
  Ainda assim, empata com o replay uniforme (84,2%), não o supera.
- **Sono** (`uniform_sleep`): os últimos 300 passos de cada bloco (10%) usam
  só a memória, sem dados novos, com o mesmo total de passos. Acerto final
  84,0%, esquecimento 1,3 ponto: igual ao replay uniforme, dentro do desvio.
  Neste orçamento, "dormir" não acrescenta nada mensurável.
- **Transformer no mesmo protocolo** (`configs/continual_tf.toml`, 1,5 M
  parâmetros, 3 seeds):

  | estratégia | acerto final médio | esquecimento médio ↓ | localização logo após treinar | posse logo após treinar |
  |---|---|---|---|---|
  | Transformer sem replay | 44,9 ± 5,7% | 16,7 ± 2,5 | 46,3% | 57,4% |
  | Transformer + replay uniforme | 52,6 ± 3,8% | 1,0 ± 1,3 | 46,9% | 52,3% |
  | nossa rede sem replay | 58,4 ± 0,5% | 40,0 ± 3,2 | 72,4% | 99,3% |
  | nossa rede + replay uniforme | 84,2 ± 2,6% | 2,0 ± 0,9 | 72,4% | 98,7% |

  O Transformer esquece menos em números absolutos porque aprende menos em
  cada bloco (posse chega a 57%, contra 99%). Com replay, os dois quase
  deixam de esquecer; a diferença final (84% contra 53%) vem da capacidade
  de aprender cada bloco com 3 mil passos, não da retenção.

## Leitura

1. **O esquecimento catastrófico acontece:** sem replay, a rede perde 59
   pontos em "posse" depois de aprender correções (99 → 41%).
2. **Rever um pouco do passado resolve quase tudo:** 25% do batch com
   exemplos guardados leva o acerto final de 58% para 84% e o esquecimento
   de 40 para 2 pontos, sem custo de tempo.
3. **A consolidação sozinha (EWC) quase não ajuda:** 58 → 61%. Proteger os
   pesos "importantes" não basta quando as tarefas usam os mesmos pesos.
4. **A versão inspirada no hipocampo NÃO superou o replay uniforme.**
   - Ficou em 81,3 ± 4,3%, contra 84,2 ± 2,6%, e esqueceu mais localização.
   - Escolher as experiências mais surpreendentes concentra a memória em
     exemplos difíceis ou ambíguos, e cobre pior o bloco como um todo.
   - A diferença está dentro de cerca de um desvio. Com 3 seeds, a conclusão
     segura é "não melhorou", não "piorou".
   - Ela também custou 70% mais tempo, por causa da gravação no SQLite, da
     releitura e do solucionador.
5. **O que a versão episódica tem a mais:**
   - a memória fica em disco, sobrevive a reinícios e pode ser auditada e
     expurgada;
   - os alvos vêm do que foi realmente observado, não de rótulos guardados.
   O ganho é de engenharia e de rastreabilidade, não de acerto.

## Limitações e divergências

- O orçamento é curto: "localização" chega só a ~72% no próprio bloco, e um
  orçamento maior pode mudar a ordem das estratégias.
- λ do EWC não foi varrido; usei um valor só, 1000.
- A fração de surpresa (metade da cota) não foi varrida.
- As execuções da seed 3 de ewc, episodic e episodic_ewc rodaram com outra
  fila em paralelo na GPU. O acerto não é afetado, mas o tempo delas está
  inflado.

## Próximos testes possíveis

- Varrer λ do EWC (100, 1000, 10000) e a fração de replay (10%, 25%, 50%).
- Repetir com um orçamento maior por bloco, para ver se a ordem das
  estratégias muda quando "localização" passa dos 72%.
- Sono com fração maior (25–50% dos passos) e com replay só dos exemplos
  mais antigos.
