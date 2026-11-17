# Roadmap

| fase | conteúdo | estado |
|---|---|---|
| 0 | inspeção do ambiente e orçamento | feito — `experiments/fase0-ambiente.md` |
| 1 | gerador procedural, solucionador simbólico, splits sem vazamento | feito — `docs/DESIGN.md` |
| 2 | GRU sem memória externa; treino e avaliação | feito — `experiments/fase2-gru.md` |
| 2b | Transformer, Mamba-2, memória rápida e iterações de arquitetura | feito (iterações 1–9) — `experiments/fase2b-comparacao.md`, `docs/ITERACOES.md` |
| 2c | curva de escala (0,5 M a 5 M parâmetros; 8 mil e 24 mil passos) | em andamento — `experiments/escala.md` |
| 3 | memória persistente (SQLite + log só de acréscimo) e recuperação explícita | implementado (`brain.memory.store`, `memory-benchmark`); falta a rodada completa com 3 seeds |
| 4 | aprendizado sequencial por blocos, replay e consolidação | feito — `experiments/fase4-continuo.md` |
| 5 | módulos e roteamento, só com justificativa experimental | a fazer |
| 6 | linguagem mais ampla com corpus licenciado | a fazer |

## Pendências

- **Escala:** seeds 2 e 3 de todos os pontos e as execuções de 5 M com 24
  mil passos (`experiments/queue_round2.txt`). A fila está pausada; para
  continuar, apague `runs/_control/pause` e rode de novo a fila, que pula o
  que já terminou e retoma o treino interrompido do mesmo passo.
- **Memória (H1):** `brain memory-benchmark` com a configuração da iteração 2
  nas 3 seeds, separando retenção nos pesos, contexto, SQLite após reinício e
  sinapses salvas em disco.
- **Iteração 8 revisada:** concatenar o código de tempo a uma parte separada
  da chave, em vez de somar.
- **Fase 4:** varrer λ do EWC e a fração de replay; orçamento maior por bloco.

## Fase 3 — regras da memória persistente

- SQLite para eventos e metadados; checkpoints continuam em arquivos.
- Campos mínimos do evento: `event_id, episode_id, project_id, entity_ids,
  relation, value, event_time, recorded_at, source_id, status, supersedes_id,
  schema_version`.
- `status ∈ {observed, user_confirmed, inferred, superseded, retracted}`;
  inferência nunca é tratada como fato confirmado.
- Ordem do evento e ordem de registro guardadas separadamente.
- Expurgo explícito: apagar remove também índices e derivados.
- Recuperação determinística por entidade e projeto. Os fatos recuperados
  entram na rede como evidência (tokens extras); a resposta não pode ser
  copiada por regra.
