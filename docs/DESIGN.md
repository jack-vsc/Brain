# Design e decisões

Registra o que foi implementado nas fases de dados, solucionador e baseline,
por quê, e onde o projeto se afastou do plano inicial. Os modelos novos e os
experimentos posteriores estão em [ARQUITETURA.md](ARQUITETURA.md) e
[ITERACOES.md](ITERACOES.md).

## O que "do zero" significa aqui

- Pesos sempre iniciados aleatoriamente (init padrão do PyTorch, seed registrada).
- Tokenizer determinístico próprio, vocabulário construído **só** a partir do split de treino.
- Nenhum LLM, tokenizer ou embedding pré-treinado em lugar algum do código.
- PyTorch, CUDA e bibliotecas de infraestrutura são usados normalmente.

## Micromundo

Léxico fixo: 12 pessoas, 12 objetos, 12 lugares, 2 projetos (`alfa`, `beta`).
Toda frase começa com `Projeto <nome>:`; projetos nunca compartilham estado.

| frase | evento | efeito |
|---|---|---|
| `Ana foi para a escola.` | MOVE | Ana está na escola |
| `Ana deixou a chave na oficina.` | PUT | chave na oficina, sem portador; Ana na oficina |
| `Ana pegou a chave.` | TAKE | Ana porta a chave; se o local da chave era conhecido, Ana está lá |
| `Ana levou a chave para casa.` | CARRY | Ana porta a chave e está em casa |
| `Ana deu a chave para Bruno.` | GIVE | Bruno porta a chave; se o local de Ana é conhecido, Bruno está lá |
| `Ana largou a chave.` | DROP | chave sem portador, no local de Ana (talvez desconhecido) |
| `Ana não levou a chave para a praia.` | negação | nenhum efeito |
| `Imagine que Ana levou a chave para a praia.` | hipótese | nenhum efeito |
| `Correção: Ana deixou a chave no parque.` | correção | troca o lugar do evento PUT mais recente de (Ana, chave) e recalcula tudo |

O local de um objeto portado é o local do portador. Correções só valem para
`foi`, `deixou` e `levou` e só trocam o **lugar**, nunca quem porta o quê,
o que mantém a história sempre consistente depois de recalculada.

### Perguntas (= tarefas reportadas separadamente)

| tarefa | exemplo | respostas possíveis |
|---|---|---|
| `loc_obj` | `onde está a chave agora?` | lugar, UNKNOWN |
| `loc_person` | `onde está Ana agora?` | lugar, UNKNOWN |
| `holder` | `quem está com a chave?` | pessoa, `ninguém`, UNKNOWN |
| `first` | `onde a chave estava primeiro?` | lugar, UNKNOWN |
| `before` | `onde a chave estava antes?` | lugar, UNKNOWN |
| `yesno` | `a chave está na oficina?` | `sim`, `não`, UNKNOWN |

UNKNOWN aparece quando a entidade nunca foi mencionada factualmente no projeto
(inclusive quando só aparece no *outro* projeto, ou só numa negação/hipótese)
ou quando o local não pode ser deduzido. `first` e `before` usam o histórico
de locais conhecidos, já com correções aplicadas.

### Tags por pergunta

`two_projects`, `cross_project` (entidade só existe no outro projeto),
`unmentioned`, `correction`, `negation` / `hypothetical` (há um distrator sobre
a própria entidade), `unknown_expected`, `heldout_pair` (só no test_comp).

## Splits e vazamento

- Splits: `train`, `val`, `test_iid`, `test_comp`. Cada episódio tem seed
  própria (`"<seed>:<split>:<tentativa>"`), e todas as perguntas de um
  episódio ficam no mesmo split.
- Nenhuma história (texto completo) aparece em dois splits; duplicatas são
  rejeitadas na geração e verificadas em teste.
- **Generalização composicional:** 15% dos pares (objeto, lugar) são
  reservados. Em train/val/test_iid esses pares nunca aparecem, nem no texto
  (inclusive em negações, hipóteses e perguntas sim/não) nem no estado do
  mundo (um objeto portado nunca passa por um lugar reservado para ele). No
  test_comp todo episódio tem ao menos uma pergunta cuja resposta é um par
  reservado. Todos os tokens do test_comp são conhecidos.
- `data_version` = hash dos arquivos gerados + versão do gerador; o
  checkpoint guarda a versão e a avaliação avisa se não conferir.

## Solucionador simbólico

`brain.symbolic.solver` faz parsing do **texto** com uma gramática fixa e
reimplementa a semântica sem reutilizar o código do gerador. Os testes exigem
100% de concordância com os rótulos do gerador, o que é uma verificação
cruzada das duas implementações. Ele é referência: nunca alimenta a rede, e
toda saída dele é rotulada "simbólico" (coluna separada na avaliação e linha
separada no chat).

## Baseline neural (Fase 2)

`embedding(128) → GRU(2 camadas, 384) → LayerNorm → Linear(30 classes)`,
~1,5 M parâmetros. Entrada: `<bos> linha <sep> linha <sep> … <q> pergunta <eos>`.
Padding tratado com `pack_padded_sequence` (testado: a saída não depende do
padding do batch). Sem memória externa: tudo o que a rede sabe sobre o
episódio precisa caber no estado recorrente.

Duas variantes com o mesmo orçamento:
- `question_position = "after"`: a pergunta só chega no fim; a rede precisa
  manter todo o estado relevante da história.
- `question_position = "before"`: a rede sabe o que acompanhar enquanto lê.

## Treino e checkpoints

- Cross-entropy + AdamW, warmup linear + cosseno, clipping de gradiente.
- Perfil de memória antes de treinar, com o batch de sequências mais longas.
  Se não couber em (VRAM livre − margem), o batch cai pela metade e o
  gradient accumulation sobe para manter o batch efetivo. CPU só como fallback
  explícito, registrado em `profile.json` e no log.
- AMP fp16 só se um passo de teste funcionar.
- Checkpoint (`last.pt`, `best.pt`): pesos, otimizador, scheduler, scaler,
  RNGs (Python e torch CPU/CUDA), passo/época/posição na época, configuração,
  vocabulário, rótulos, versão dos dados e métricas de validação. Escrita em
  `.tmp` + `os.replace`. Leitura com `torch.load(weights_only=True)` (testado
  contra um pickle malicioso).
- Retomada: a ordem dos exemplos é uma permutação determinística por época;
  ao retomar, os exemplos já vistos naquela época são pulados. Os
  `DataLoader`s usam gerador próprio para não consumir o RNG global. Testado
  em CPU: interromper e retomar dá os mesmos pesos finais que o treino
  contínuo. Na GPU, o cuDNN pode introduzir pequenas diferenças não
  determinísticas.
- Orçamento de tempo (`max_minutes`), early stopping por paciência e Ctrl+C
  salvam `last.pt` antes de sair.
- `--pilot-steps N` roda N passos sem salvar checkpoint e grava em
  `pilot.json` uma estimativa de custo do treino completo.

## Desvios do plano inicial

1. **Pacote em `src/brain/<subpacote>`** em vez de `src/<subpacote>`:
   nomes como `data` e `models` como pacotes de topo colidem facilmente com
   outras bibliotecas. Os subdiretórios previstos existem dentro de
   `src/brain/`. O solucionador ficou em `src/brain/symbolic/`.
2. **Ambiente fora do disco do sistema.** A máquina de desenvolvimento tinha
   pouco espaço no C:, e PyTorch com CUDA ocupa cerca de 3 GB, então a venv
   ficou em outro disco. Nada no código depende disso.
3. **Classes extras na saída**: `sim`, `não` e `ninguém`, além de
   entidades/lugares/UNKNOWN, para cobrir perguntas sim/não e posse vazia.
4. **Negação só como distrator**: "não …" nunca altera o estado. Fatos
   negativos ("a chave não está na escola") com efeito em perguntas sim/não
   ficaram para depois.
5. **Mundo "frouxo"**: pegar um objeto teletransporta a pessoa para o local
   do objeto, se conhecido; não há restrição de co-presença no texto. É uma
   regra explícita, aplicada igualmente pelo gerador e pelo solucionador.
6. **GRU multicamada própria** (`StackedGRU`) em vez de
   `nn.GRU(num_layers=2, dropout>0)`: com torch 2.14.1+cu130 no Windows, o
   estado de dropout do cuDNN derruba o processo ao sair. Camadas simples com
   `nn.Dropout` entre elas fazem a mesma conta e carregam checkpoints antigos.
