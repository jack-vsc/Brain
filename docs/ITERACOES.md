# Diário de iterações de arquitetura

Objetivo: uma rede "o mais próxima de um cérebro" que supere os
Transformers. Regra deste diário: cada iteração tem hipótese, mecanismo e
resultado **medido**, com o mesmo orçamento (~1,5 M parâmetros, 8 mil passos,
batch 64, mesmos dados `mvp2`, pergunta depois da história), lr escolhido
**só pela validação** e 3 seeds antes de qualquer conclusão. "Inspirado no
cérebro" descreve o mecanismo; não é uma afirmação biológica.

Baselines: GRU (lr 1e-3), Transformer pre-LN com RoPE (lr 5e-4, o melhor de
{3e-4, 5e-4, 1e-3, 2e-3}) e Mamba-2 do zero (lr 1e-3, o melhor de {5e-4, 1e-3,
2e-3} pela validação). Mamba-2, 3 seeds: test_iid 66,1 ± 1,8%, pares inéditos
16,6 ± 5,3%, histórias longas 54,6 ± 1,2%, isolamento 18,4 ± 2,8%. Fica acima
do Transformer em test_iid e nas histórias longas, mas muito abaixo da
memória rápida, e usa mais VRAM (1,55 GB no treino).

## Iteração 1 — memória rápida com regra delta

- **Hipótese:** o gargalo da GRU é guardar tudo num único vetor de estado
  (com a pergunta antes, ela vai de 55% para 83%). Uma memória associativa
  que a rede escreve enquanto lê resolveria isso.
- **Mecanismo:** matriz W por episódio; a cada token, chave/valor/força
  vindos da GRU; `W ← W + β k (v − Wᵀk)ᵀ` (substitui o valor antigo daquela
  chave). Leitura no fim com uma consulta.
- **Inspiração:** fast weights (Schmidhuber 1992; Ba et al. 2016), DeltaNet.
  No cérebro: ligações sinápticas rápidas, de curta duração.
- **Engenharia:** forma em blocos exata (sistema triangular), 4,2x mais rápida.
- **Resultado (3 seeds):** test_iid 86,5 ± 1,1% contra 62,5 ± 0,8% (GRU) e
  58,3 ± 2,2% (Transformer). Isolamento instável entre seeds (68 ± 22%).

## Iteração 2 — passos internos de leitura ("pensar antes de responder")

- **Hipótese:** perguntas encadeadas (objeto → portador → local do portador)
  precisam de mais de uma leitura.
- **Mecanismo:** 3 leituras; cada uma atualiza o estado de controle e gera a
  próxima consulta.
- **Inspiração:** computação adaptativa / recorrência em profundidade ("K passos
  internos" antes de responder).
- **Resultado (3 seeds):** test_iid 89,9 ± 1,2%, pares inéditos 77,2 ± 1,3%
  (Transformer 30,2 ± 8,0%), histórias longas 78,2 ± 2,0% (Transformer
  47,0 ± 1,9%), isolamento 87,8 ± 5,0%, respostas inventadas 8,2%. Com
  orçamento 3x, o Transformer chega a 79,8% em test_iid; a iteração 2 a 92,5%.
- **Ainda fraco:** "onde estava antes?" (~75%) e histórias longas.

## Iteração 3 — separação de padrões + familiaridade

- **Hipótese H5:** chaves grandes e esparsas reduzem a interferência entre
  memórias parecidas (mesmo objeto em projetos diferentes), e um sinal
  explícito de "força da leitura" ajuda a responder UNKNOWN.
- **Mecanismo:** chaves/consultas de 128 dimensões com só 16 ativas (top-k);
  a norma da leitura de cada cabeça vai para a cabeça de saída.
- **Inspiração:** giro denteado (codificação esparsa e expandida), sinal de
  familiaridade do córtex perirrinal.
- **Resultado (seed 1): FALHOU.** Val 75,1% (iteração 2: 90,7%). O
  isolamento, que era o alvo, piorou de 91% para 23%, e "onde está a pessoa?"
  caiu para 45%.
- **Diagnóstico:** falha de otimização, não de generalização. A perda de
  treino empacou em 0,49 (iteração 2: 0,06), com a validação estabilizada
  desde ~5 mil passos. O top-k duro deixa gradiente em só 16 de 128
  dimensões por chave e favorece unidades mortas (ReLU), então a rede não
  aprende bons endereços. A familiaridade não foi testada isoladamente
  (efeito confundido).
- **Conclusão:** esta implementação de separação de padrões está descartada;
  a ideia não. Próximas tentativas possíveis: esparsidade suave (k-WTA com
  straight-through, temperatura, penalidade L1) e familiaridade em ablação
  separada. Confirmado nas 3 seeds: test_iid 74,8 ± 3,1%, isolamento
  20,7 ± 3,2%.

## Iteração 4 — memória episódica nas fronteiras de eventos, com reinstauração

- **Hipótese H6:** a rede só consegue guardar "onde estava antes" se puder
  **ler** o valor antigo no momento em que escreve o novo.
- **Mecanismo:** a GRU lê frase por frase; no fim de cada frase consulta a
  memória, reinjeta a leitura no estado (reinstauração) e só então escreve.
  Pergunta respondida com 3 passos de leitura e chaves densas (como na
  iteração 2; chaves esparsas e familiaridade foram retiradas depois da falha
  da iteração 3). Ablação: a mesma rede sem reinstauração.
- **Inspiração:** teoria de segmentação de eventos (o hipocampo codifica nas
  fronteiras de eventos); reinstauração de contexto entre hipocampo e córtex.
- **Engenharia:** laço só sobre frases (cuDNN dentro de cada frase), com
  equivalência exata a um laço token a token verificada em teste.
- **Resultado (seed 1): FALHOU, e a comparação planejada ficou confundida.**
  - Sem reinstauração (chaves densas, mesma base da iteração 2): val 70,0%
    (iteração 2: 89,5 ± 1,6%), perda de treino 0,48.
  - Com reinstauração: val 67,6%, mas este treino começou antes da troca de
    configuração e rodou com chaves esparsas + familiaridade (os componentes
    da iteração 3, que falhou). **Não vale como ablação.**
  - As duas rodaram em paralelo com outras filas (erro de encadeamento meu),
    então os tempos estão inflados; a acurácia não é afetada.
- **O que dá para concluir:** escrever na memória **só no fim de cada frase**
  é muito pior do que escrever a cada token (−20 pontos, com subajuste). Uma
  escrita por frase não basta para frases que mudam vários fatos de uma vez
  ("Ana deu a chave para Bruno" muda o portador da chave e o local de Bruno).
  A segmentação por eventos, nesta forma, atrapalha.

## Iteração 5 — iteração 2 + reinstauração (teste limpo de H6)

- **Mecanismo:** escrita a cada token, exatamente como na iteração 2; no fim
  de cada frase, a rede lê a memória e reinjeta a leitura no estado antes da
  frase seguinte (sem escrever nesse momento).
- **Ablação:** a mesma rede com `reinstate=false`. Mesmo número de parâmetros
  (1,50 M), mesma conta da iteração 2. 3 seeds em cada braço.
- **Engenharia:** equivalência com um laço token a token verificada em teste,
  com e sem reinstauração.
- **Resultado (3 seeds cada; tabela em `experiments/iter5-tabela.md`): a
  reinstauração FALHOU (H6 refutada neste micromundo).**
  - Com reinstauração: test_iid 88,3 ± 4,5%, pares inéditos 73,9 ± 6,1%,
    histórias longas 74,6 ± 4,9%, isolamento 68,7 ± 41,1%. Uma das seeds
    perdeu quase todo o isolamento entre projetos.
  - Sem reinstauração (ablação, mesmos parâmetros): test_iid 91,2 ± 0,3%,
    pares inéditos 80,1 ± 0,7%, histórias longas 78,4 ± 0,3%, isolamento
    92,1 ± 1,9%.
  - Ler a memória e reinjetar a leitura entre as frases piora a média e deixa
    o treino instável. A hipótese era que isso ajudaria a ligar fatos
    distantes, mas a rede já faz isso com as 3 leituras do fim.
- **Achado lateral:** o braço sem reinstauração tem a mesma conta da
  iteração 2, implementada com o laço por frases. Ele fica um pouco acima da
  iteração 2 (91,2 ± 0,3% contra 89,9 ± 1,2%) e com desvio bem menor. A
  diferença está perto de um desvio da iteração 2; não trato como melhoria
  de arquitetura, só como sinal de que o laço por frases não atrapalha.
- **Custo:** o laço por frases é cerca de 3 vezes mais lento por passo
  (0,17–0,20 s contra 0,06 s).

## Iteração 6 — leituras adaptativas ("pensar só o necessário")

- **Hipótese:** perguntas encadeadas precisam de mais leituras da memória e
  as simples de menos; deixar a rede decidir quando parar economiza passos
  sem perder acerto.
- **Mecanismo:** até 5 leituras; depois de cada uma, a rede estima a
  probabilidade de parar (estilo PonderNet). A resposta é a média ponderada
  pelas probabilidades de parada, e um custo de 0,01 × passos esperados entra
  na perda.
- **Inspiração:** computação adaptativa; o cérebro gasta mais tempo em
  decisões difíceis.
- **Resultado (3 seeds; tabela em `experiments/iter67-tabela.md`): FALHOU.**
  test_iid 86,4 ± 1,0% (iteração 2: 89,9 ± 1,2%), pares inéditos
  67,2 ± 2,5% (77,2%), histórias longas 74,1 ± 1,3% (78,2%), isolamento
  64,4 ± 18,4%.
- **O que a rede aprendeu a fazer:** usa em média 1,7 das 5 leituras. As
  perguntas de portador ("quem está com…?"), que são encadeadas, ficaram com
  o MENOR número de leituras (1,2–1,5), e as de local do objeto com o maior
  (cerca de 2,0). É o oposto da hipótese: o custo empurrou a rede a parar
  cedo, e ela perdeu justamente o raciocínio em várias etapas que as 3
  leituras fixas davam.
- **Próximo teste possível:** custo menor (0,001) ou um mínimo de 2 leituras.

## Iteração 7 — porta de esquecimento na memória rápida

- **Hipótese:** deixar a memória esquecer aos poucos (Gated DeltaNet) libera
  espaço e ajuda nas histórias longas.
- **Mecanismo:** antes de cada escrita, W é multiplicada por α ∈ (0, 1) por
  cabeça, calculado a cada token.
- **Resultado (3 seeds): FALHOU feio.** test_iid 69,2 ± 4,9%, pares inéditos
  37,9 ± 10,4%, histórias longas 56,3 ± 5,1%, isolamento 35,7 ± 21,4%, e
  respostas inventadas em 34% dos casos.
- **Causa provável (não comprovada): a inicialização.** α começa em ≈ 0,95
  por token. Numa história de 60 tokens, isso guarda só 0,95⁶⁰ ≈ 5% de um
  fato do começo. As perguntas deste micromundo dependem de fatos de
  qualquer ponto da história, e a rede não aprendeu a abrir a porta.
- **Nota de medição:** a partir de 25/10 às 17h25, duas filas rodam ao mesmo
  tempo na GPU, com uso subindo de cerca de 40% para cerca de 92%. Cada
  treino fica mais lento (iteração 7b: 0,108 s/passo contra 0,063 s sozinho),
  mas o conjunto anda cerca de 45% mais rápido. O acerto não muda. Tempo por
  passo, tempo de treino e latência das execuções desse período (7b,
  Mamba-2 seeds 2–3, Fase 4 seed 3 de ewc/episodic/episodic_ewc) não servem
  para comparar velocidade com as anteriores.
- **Iteração 7b (3 seeds; tabela em `experiments/iter7b-tabela.md`):** a mesma
  rede com α ≈ 0,9975 no início (≈ 86% retido em 60 tokens).
  - test_iid 89,5 ± 0,1%, pares inéditos 75,5 ± 1,2%, histórias longas
    76,9 ± 0,8%, isolamento 85,9 ± 3,3%.
  - **A causa estava na inicialização:** só mudar o viés levou de 69% para
    89,5%.
  - **Mas a porta não traz ganho:** fica igual à iteração 2 em test_iid
    (89,9 ± 1,2%) e um pouco abaixo nas histórias longas (78,2%). Neste
    micromundo, qualquer fato antigo pode ser perguntado, então esquecer não
    ajuda. Volta a ser candidata com histórias bem mais longas que a
    memória.

## Iteração 8 — código de tempo na memória ("células de tempo")

- **Hipótese:** "onde estava antes?" é a pergunta mais fraca (~75%) porque a
  memória rápida guarda o quê, mas não o quando.
- **Mecanismo:** o índice da frase de cada token vira senos e cossenos em 5
  escalas (1, 2, 4, 8 e 16 frases). Uma projeção aprendida soma esse código à
  chave e ao valor de cada escrita. +4,2 mil parâmetros (1,506 M no total).
- **Inspiração:** células de tempo do hipocampo, que disparam em momentos
  sucessivos de uma experiência.
- **Resultado (3 seeds; tabela em `experiments/iter89-tabela.md`): FALHOU.**
  test_iid 78,6 ± 1,1% (iteração 2: 89,9 ± 1,2%), pares inéditos
  51,5 ± 1,4%, histórias longas 64,8 ± 0,9%, isolamento 21,4 ± 0,4% e
  respostas inventadas em 26% dos casos.
- **Nem o alvo melhorou:** "onde estava antes?" ficou em 69,8 ± 1,8%, abaixo
  dos 75,3% da iteração 2.
- **Leitura provável:** o código de tempo é o mesmo nos dois projetos de uma
  história, então somar ele à chave aproxima justamente as memórias que
  precisavam ficar separadas. O isolamento entre projetos despencou (88% →
  21%), e o resto caiu junto. Uma versão que concatene o tempo a uma parte
  separada da chave, em vez de somar, seria o próximo teste.

## Iteração 9 — minGRU (recorrência paralela)

- **Hipótese:** trocar a GRU pela minGRU (Feng et al., 2024), cuja porta não
  depende do estado anterior e que pode ser calculada em paralelo, acelera o
  treino sem perder acerto.
- **Mecanismo:** 3 camadas minGRU de 470 unidades com LayerNorm (1,496 M
  parâmetros), resolvidas em espaço log com `logcumsumexp`. A equivalência
  com o laço token a token foi verificada em teste.
- **Medida antes do treino:** com histórias curtas (~80 tokens), um passo de
  ida e volta levou 65 ms, contra 57 ms da GRU do cuDNN, então a esperada
  aceleração não apareceu nesse comprimento. Medido com a GPU ocupada por
  outro treino. No piloto de 300 passos, a validação chegou a 49%, contra 38%
  da iteração 8.
- **Resultado (3 seeds; tabela em `experiments/iter89-tabela.md`): empate
  com a iteração 2, só que instável.** test_iid 87,5 ± 4,0%, pares inéditos
  74,2 ± 5,6%, histórias longas 75,3 ± 4,7%. "Onde estava antes?" ficou em
  75,9%, igual. O isolamento variou de ~10% a ~95% entre seeds (44 ± 45%).
- **Custo:** latência por resposta um pouco menor (7,7 ms contra 9,4 ms), mas
  o treino não ficou mais rápido e a VRAM subiu (903 MB contra 352 MB): a
  varredura em espaço log materializa tensores grandes. Com histórias deste
  tamanho, a GRU do cuDNN continua sendo a melhor escolha.
- **Nota de medição:** as iterações 8 e 9 rodaram com duas filas ao mesmo
  tempo na GPU; os tempos de treino delas não servem para comparar velocidade.

## Resumo até aqui

A iteração 2 (GRU + memória rápida com regra delta + 3 leituras) continua
sendo a melhor configuração: 89,9% em test_iid contra 58,3% do Transformer com
o mesmo orçamento. Das ideias testadas depois dela (iterações 3 a 9), nenhuma melhorou o
acerto; duas empataram (porta de esquecimento iniciada aberta e minGRU) e as
demais pioraram. O curto prazo da rede está bem resolvido para este
micromundo; o que falta testar exige histórias bem mais longas ou outro tipo
de tarefa.
