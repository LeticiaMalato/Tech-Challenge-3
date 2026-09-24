# Sistema de Triagem Automática de Urgência — Tech Challenge Fase 3

Sistema de classificação de urgência de laudos médicos (`normal` / `attention` / `urgent`), servido via API REST, com pipeline completo de MLOps: dados → treino → API → CI/CD → orquestração → monitoramento.

## Sumário

- [Contexto do problema](#contexto-do-problema)
- [Arquitetura](#arquitetura)
- [Decisão de arquitetura em nuvem](#decisão-de-arquitetura-em-nuvem)
- [Dataset e preparação de dados](#dataset-e-preparação-de-dados)
- [Modelo](#modelo)
- [Resultados e limitações conhecidas](#resultados-e-limitações-conhecidas)
- [API](#api)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Orquestração (Airflow)](#orquestração-airflow)
- [Monitoramento](#monitoramento)
- [Latência](#latência)
- [Como executar](#como-executar)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Decisões de design registradas](#decisões-de-design-registradas)
- [Próximos passos](#próximos-passos)

---

## Contexto do problema

Um hospital de referência precisa de um sistema de triagem automática de laudos médicos, classificando a urgência de cada caso para priorizar atendimento. O foco deste projeto não é apenas a acurácia do modelo, mas o ciclo de vida completo de um sistema de ML em produção: pipeline de dados, API, containerização, CI/CD, orquestração de retreino e observabilidade.

## Arquitetura

```
                    ┌───────────────┐
                    │    Cliente    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    FastAPI    │──── /health, /ready, /metrics
                    │   /predict    │
                    └───────┬───────┘
                            │
                     ┌──────┴──────┐
                     ▼             ▼
              TfidfPreprocessor  OnnxClassifier
                     │             │
                     └──────┬──────┘
                            ▼
                       Prometheus ──► Grafana (3 painéis)

GitHub ──► GitHub Actions (lint → test → build)

Airflow DAG: prepare_data.py >> train.py ──► model_artifacts/ (joblib + ONNX)
```

A API, o Prometheus e o Grafana rodam como serviços de um mesmo `docker-compose.yml`, com sequenciamento de inicialização baseado em healthchecks (`api` saudável → `prometheus` saudável → `grafana` sobe).

## Decisão de arquitetura em nuvem

**Modelo de deploy: real-time, não batch.**

Triagem hospitalar é, por natureza, uma operação **por requisição individual**: cada laudo chega isoladamente e precisa de uma resposta imediata para orientar a fila de atendimento — não existe um cenário em que "esperar o próximo lote horário" seja aceitável clinicamente. O próprio baseline de latência medido (mediana ~4ms, P95 ~7ms por requisição) confirma que o modelo é leve o suficiente para servir em tempo real sem sacrificar throughput.

**Provedor recomendado: AWS**, pelos seguintes motivos, mapeados diretamente aos componentes já construídos neste projeto:

| Componente já construído | Serviço AWS equivalente | Por quê |
|---|---|---|
| `Dockerfile` da API | **ECS Fargate** | Roda o container sem gerenciar servidores; escala horizontalmente conforme demanda de requisições, sem mudar uma linha do `Dockerfile` já existente. |
| `docker-compose.yml` (build local) | **ECR** (registro de imagens) | Armazena a imagem construída pelo CI, para o ECS puxar em produção. |
| `model_artifacts/*.joblib` e `classifier.onnx` | **S3** | Armazenamento durável dos artefatos treinados; o `lifespan` da API poderia ser adaptado para baixar do S3 em vez de `COPY` na imagem, desacoplando deploy de código de deploy de modelo. |
| `dags/training_pipeline_dag.py` | **MWAA** (Managed Workflows for Apache Airflow) | Roda a DAG já escrita sem precisar operar infraestrutura própria de Airflow (scheduler, banco, webserver). |
| Prometheus + Grafana | **Amazon Managed Service for Prometheus/Grafana** (ou EC2 auto-hospedado, como neste projeto) | Path de menor atrito para manter a mesma stack de observabilidade já validada, com opção gerenciada se o time preferir reduzir operação. |
| `ci.yml` (GitHub Actions) | Mantido como está | GitHub Actions já se integra nativamente com ECR/ECS via `aws-actions`, sem necessidade de trocar de ferramenta de CI. |

**Por que não batch:** processamento em lote (ex.: um job noturno reclassificando laudos acumulados) faria sentido para relatórios agregados ou auditoria retroativa — não para o caso de uso central deste desafio, que é priorização de atendimento em tempo real.

Esta decisão foi tomada apenas ao final do desenvolvimento, deliberadamente — como registrado ao longo do projeto, evitar comprometer-se com uma arquitetura de nuvem antes de ter números reais de latência (medidos apenas na Etapa 1) seria decidir sem evidência.

## Dataset e preparação de dados

**Fonte:** [Medical Abstracts TC Corpus](https://huggingface.co/datasets/sebischair/Medical-Abstracts-TC-Corpus) (`sebischair/Medical-Abstracts-TC-Corpus`), 5 categorias de condição médica.

**Mapeamento de urgência** (`scripts/prepare_data.py`):

| Categoria original | Urgência mapeada |
|---|---|
| `cardiovascular diseases` | `urgent` |
| `nervous system diseases` | `urgent` |
| `neoplasms` | `attention` |
| `digestive system diseases` | `attention` |
| `general pathological conditions` | `normal` |

> **Limitação de modelagem, documentada explicitamente:** este mapeamento é uma simplificação categoria→urgência. Um caso de câncer em estágio avançado (`neoplasms`) sempre mapeia para `attention`, nunca `urgent`, independentemente da gravidade real descrita no texto — a granularidade do dataset original não permite diferenciar estágio/gravidade dentro de uma mesma categoria de doença. Da mesma forma, `nervous system diseases` mistura emergências neurológicas com procedimentos ortopédicos de rotina (ex.: descompressão de nervo ulnar), o que se mostrou uma fonte real de erro do modelo (ver seção de limitações abaixo).

**Achado crítico de qualidade de dados:** a EDA revelou que, do pool combinado de `train` + `test` originais (14.438 linhas), **32% dos textos tinham o mesmo abstract associado a rótulos de urgência conflitantes** em ocorrências diferentes (o mesmo laudo aparecendo como `urgent` em uma linha e `normal` em outra) — incluindo **35% de overlap direto entre os splits originais de train/test**, caracterizando vazamento de dados.

**Resolução:**
1. União do pool `train` + `test` original.
2. Remoção de todos os textos com rótulo de urgência conflitante entre suas ocorrências (mantendo apenas textos com rótulo consistente).
3. Deduplicação dos textos remanescentes.
4. Reconstrução de um split próprio 70/15/15 (`train`/`val`/`test`), estratificado por `urgency_label`, com `random_state=42` fixo para reprodutibilidade.

**Dataset final:** 8.546 amostras (`train`: 5.982 / `val`: 1.282 / `test`: 1.282), com distribuição de classes moderadamente desbalanceada (`urgent` 36,6% / `attention` 35,4% / `normal` 28,0%).

## Modelo

**TF-IDF + Logistic Regression**, escolhido deliberadamente em vez de Random Forest (sugestão do enunciado):

- Modelos lineares performam melhor com features TF-IDF de alta dimensionalidade e esparsas (~16 mil termos no vocabulário).
- Significativamente mais leve e rápido na inferência — alinhado ao requisito central de baixa latência do desafio.
- `StandardScaler` **não é aplicado**: normalização padrão destruiria a esparsidade da matriz TF-IDF, inflando uso de memória sem ganho de performance (o TF-IDF já aplica normalização L2 internamente).

**Hiperparâmetros validados via EDA:**
- `TfidfVectorizer(min_df=2, stop_words="english")`
- `LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)` — `class_weight="balanced"` justificado pelo desbalanceamento moderado observado na EDA.

**Inferência em produção: ONNX Runtime.** O treino continua em scikit-learn; `scripts/train.py` exporta o `LogisticRegression` para `model_artifacts/classifier.onnx` (`skl2onnx`). A API carrega esse grafo via `OnnxClassifier` (uma sessão, labels + probabilidades juntos). O TF-IDF permanece em sklearn: a matriz esparsa (~16 mil termos) só é densificada na entrada do ONNX. Os `.joblib` ficam para retreino e para o benchmark sklearn vs ONNX.

**Métrica principal: macro F1**, não accuracy — escolhida porque trata as 3 classes com peso igual, relevante tanto pelo desbalanceamento quanto pelo custo assimétrico de erros em triagem médica (confundir `urgent` com `normal` é clinicamente muito mais grave que confundir `attention` com `normal`).

## Resultados e limitações conhecidas

**Baseline (`scripts/train.py`):** macro F1 = **0,813** no conjunto de validação.

| Classe | F1-score |
|---|---|
| `attention` | 0,89 |
| `urgent` | 0,83 |
| `normal` | 0,72 |

A classe `normal` é a mais fraca, consistente com a EDA (vocabulário menos distintivo, sem termos técnicos específicos que a diferenciem claramente).

**Análise de erro (`notebook/error_analysis.ipynb`):** investigação quantificada, usando o conjunto de validação real (não exemplos hipotéticos), focada no erro clinicamente mais grave — casos verdadeiramente `urgent` classificados como `normal`.

- **16,2% dos casos verdadeiramente `urgent`** no conjunto de validação (76 de 469) são classificados incorretamente como `normal`.
- **Causa raiz identificada:** a categoria `nervous system diseases` (mapeada para `urgent`) mistura emergências neurológicas com procedimentos ortopédicos/administrativos de rotina. Inspeção dos coeficientes do `LogisticRegression` mostrou que termos como "sudden" e "emergency" — presentes em casos reais de erro — têm peso aprendido favorecendo a classe `normal`, provavelmente por associação estatística no corpus de treino (artigos científicos), que difere do uso clínico real dessas palavras em contexto de triagem.
- **Limitação de fundo:** modelos lineares bag-of-words não capturam contexto semântico — dependem inteiramente do vocabulário estatístico do corpus de treino, sem generalização para sinônimos ou uso contextual diferente.

## API

FastAPI com 4 endpoints:

| Endpoint | Método | Propósito |
|---|---|---|
| `/predict` | `POST` | Recebe `{"text": "..."}`, devolve `{"urgency": "...", "probabilities": {...}}` |
| `/health` | `GET` | Liveness check — confirma apenas que o processo está de pé, sem depender do modelo |
| `/ready` | `GET` | Readiness check — confirma que `preprocessor`/`classifier` foram carregados (`503` caso contrário) |
| `/metrics` | `GET` | Métricas Prometheus (formato texto) |

**Validação de entrada:** `text` exige `min_length=1` (rejeita string vazia com `422`).

**Tratamento de erro:** falhas internas de processamento (`transform`/`predict`) retornam `400` com mensagem clara, em vez de vazar stacktrace como `500`.

**Carregamento de modelo:** artefatos (`preprocessor.joblib`, `classifier.onnx`) são carregados uma única vez, no `lifespan` da aplicação — não a cada requisição. Consequência operacional: atualizar o modelo em produção requer reiniciar o serviço (reconstruir a imagem Docker); não há hot-reload dinâmico, por decisão de escopo.

## Docker

- Imagem baseada em `python:3.12-slim`, alinhada à versão de Python já validada em desenvolvimento.
- Dependências instaladas via `uv sync --frozen`, garantindo reprodutibilidade exata das versões do `uv.lock`.
- Grupo `dev` (pytest, ruff, pre-commit) **excluído** da imagem de produção (`--no-dev`), tanto no build quanto no `CMD` de execução.
- Container roda como usuário não-root (`appuser`), com `chown -R` garantindo posse correta dos arquivos.
- `HEALTHCHECK` usando o próprio endpoint `/health`, via `python -c "..."` (evita dependência de `curl`, ausente na imagem `slim`).
- Artefatos do modelo são copiados **para dentro da imagem** no momento do build (estratégia "bake", não volume externo) — retreinar implica reconstruir a imagem, consistente com a decisão de não implementar hot-reload.

## CI/CD

`.github/workflows/ci.yml`, dois jobs:

1. **`lint-and-test`**: checkout → instala `uv` (com cache) → `uv sync --frozen` → `ruff check .` → `pytest tests/ -v`.
2. **`docker-build`**: depende de `lint-and-test` (via `needs:`), evitando build desnecessário se lint/testes falharem. Valida que a imagem Docker constrói com sucesso.

Gatilhos em `push`/`pull_request` para `main` e `dev` — feedback cedo a cada PR de feature, não só no merge final.

**Decisão de reprodutibilidade:** `model_artifacts/*.joblib` e `classifier.onnx` são **versionados no Git**, para que o job `docker-build` no runner do GitHub Actions (ambiente limpo, sem dataset bruto) tenha o que copiar na imagem. Consequência: re-treinos locais exigem commitar os artefatos atualizados, ou o CI (e qualquer clone do repositório) usa o modelo desatualizado.

**Testes:** 6 testes automatizados — `TfidfPreprocessor.transform` (formato/tipo), `/predict` (sucesso e `422`, usando fakes de preprocessor/classifier para desacoplar dos artefatos reais), `remove_conflicting_labels` (conflito removido, duplicata consistente deduplicada, texto único mantido), e paridade sklearn vs ONNX (rótulos iguais e probabilidades a < 1e-5).

## Orquestração (Airflow)

`dags/training_pipeline_dag.py`: DAG de 2 tasks (`BashOperator`), orquestrando os scripts já existentes sem duplicar lógica:

```
prepare_data_task >> train_model_task
```

- `schedule=None`: disparo manual, não automático — reflete a natureza do dataset (corpus estático, sem chegada contínua de dados que justifique agendamento periódico).
- `retries=1` em ambas as tasks.

**Limitações de execução, documentadas conscientemente:**
- A DAG assume que o worker do Airflow tem acesso ao mesmo ambiente Python do projeto (dependências instaladas, execução a partir da raiz do repositório) — não implementado nesta entrega.
- O ambiente completo do Airflow (scheduler, webserver, banco de dados via Docker Compose) **não foi executado de ponta a ponta** — decisão de escopo, já que o entregável explícito da Etapa 2 é o arquivo `.py` da DAG (estrutura e sintaxe corretas, validadas), não necessariamente a infraestrutura completa rodando.
- Sem versionamento/promoção de modelo: cada execução sobrescreve `model_artifacts/classifier.joblib` e `classifier.onnx` diretamente. Uma versão mais robusta salvaria artefatos com timestamp/run_id e só promoveria o novo modelo se o F1 de validação fosse ≥ ao anterior — registrado como melhoria futura.

## Monitoramento

`docker-compose.yml` sobe 3 serviços: `api`, `prometheus`, `grafana`, com sequenciamento por healthcheck (`api` saudável → `prometheus` saudável → `grafana` sobe).

**Instrumentação (`src/app/api/metrics.py`):**
- `Counter` (`predict_requests_total`), com label `status` diferenciando `success`, `error_400` (falha interna) e `error_422` (payload inválido).
- `Histogram` (`predict_request_duration_seconds`), medindo duração de toda requisição (sucesso ou erro).
- Instrumentado via **middleware HTTP**, não dentro do endpoint — garante que erros de validação do Pydantic (que nunca chegam a executar o corpo de `/predict`) também sejam contabilizados.

**Dashboard Grafana** (`monitoring/grafana/provisioning/dashboards/triage_dashboard.json`), 3 painéis, carregados **automaticamente** via provisioning (sem configuração manual):

1. **Total Requests** — `predict_requests_total`, por status.
2. **Average Latency** — `rate(predict_request_duration_seconds_sum[5m]) / rate(predict_request_duration_seconds_count[5m])`.
3. **Error Rate** — `sum(rate(predict_requests_total{status=~"error.*"}[5m])) / sum(rate(predict_requests_total[5m]))`.

Datasource Prometheus provisionado com UID fixo (`prometheus-datasource`), evitando o problema comum de dashboards exportados manualmente ficarem presos a um ID de datasource específico da instância que os gerou.

## Latência

**Comparação sklearn vs ONNX** (`scripts/compare_latency.py`): inferência in-process, sem HTTP, no mesmo texto de amostra da medição HTTP, 20 warm-ups + 200 corridas. Isola o classificador e também o caminho completo (TF-IDF + classificador), que é o que a API executa.

| Caminho | Mean | Median | P95 |
|---|---|---|---|
| sklearn classificador (`predict` + `predict_proba`) | 0,215 ms | 0,206 ms | 0,296 ms |
| ONNX classificador (uma sessão) | 0,028 ms | 0,025 ms | 0,038 ms |
| sklearn TF-IDF + classificador | 0,508 ms | 0,507 ms | 0,667 ms |
| TF-IDF + ONNX (produção) | 0,331 ms | 0,317 ms | 0,455 ms |

- Paridade de rótulos sklearn vs ONNX: **100%** nas 200 inferências.
- Speedup mediano **só do classificador: 8,12×**. End-to-end: **1,60×** — o TF-IDF em sklearn continua sendo a maior fatia do tempo; o ganho do ONNX aparece inteiro no classificador.
- Os `.joblib` permanecem para retreino e para este benchmark; a API de produção usa só o ONNX.

**HTTP `/predict`** (`scripts/measure_latency.py`) — 50 requisições, warm-up de 5, cliente reutilizado (keep-alive):

| | Mean | Median | P95 |
|---|---|---|---|
| Baseline Etapa 1 (sklearn, sem ONNX) | 5,13 ms | 3,98 ms | 6,77 ms |
| Etapa 4 (TF-IDF + ONNX na API) | 1,99 ms | 1,85 ms | 2,72 ms |

A queda HTTP mistura o classificador mais rápido com variação de máquina/carga; o número que isola a otimização da Etapa 4 é a tabela in-process acima.

*(Nota: uma medição inicial da Etapa 1, sem warm-up e sem keep-alive, registrou valores ~8× maiores — Mean 39,78 ms / P95 60,14 ms — overhead de TCP, não do modelo.)*

## Como executar

**Pré-requisitos:** Python 3.12, [uv](https://docs.astral.sh/uv/), Docker Desktop.

```powershell
# 1. Instalar dependências
uv sync

# 2. Configurar PYTHONPATH (necessário para os imports de scripts/notebooks)
# Crie um arquivo .env na raiz com:
# PYTHONPATH=src

# 3. Preparar os dados (gera data/processed/{train,val,test}.csv)
uv run python scripts/prepare_data.py

# 4. Treinar o modelo (gera model_artifacts/*.joblib e classifier.onnx)
uv run python scripts/train.py

# 4b. Comparar latência sklearn vs ONNX (não precisa da API no ar)
uv run python scripts/compare_latency.py

# 5. Rodar a API localmente
uv run uvicorn app.api.main:app --app-dir src

# 6. OU subir a stack completa (API + Prometheus + Grafana)
docker compose up --build
```

**Acessos, com a stack rodando:**
- API: http://127.0.0.1:8000/docs
- Prometheus: http://127.0.0.1:9090
- Grafana: http://127.0.0.1:3000 (login `admin`/`admin`; dashboard já provisionado automaticamente)

**Rodar testes e lint:**
```powershell
uv run ruff check .
uv run pytest tests/ -v
```

## Estrutura do repositório

```
Tech-Challenge-3/
├── src/app/                 # Pacote de produção (vai para a imagem Docker)
│   ├── core/interfaces.py       # Contratos abstratos (Strategy pattern)
│   ├── preprocessing/            # TfidfPreprocessor
│   ├── models/                   # LogisticClassifier (treino) + OnnxClassifier (API)
│   └── api/                      # FastAPI: main.py, schemas.py, metrics.py
├── scripts/                  # Uso único, fora do pacote de produção
│   ├── prepare_data.py
│   ├── train.py
│   ├── compare_latency.py
│   └── measure_latency.py
├── dags/                     # DAG do Airflow
├── tests/                    # Testes automatizados (pytest)
├── notebook/                  # EDA e análise de erro (exploração, não produção)
├── monitoring/                # Configuração Prometheus/Grafana
├── .github/workflows/         # CI/CD
├── data/processed/            # Datasets processados (gerados por prepare_data.py)
├── model_artifacts/           # Modelo treinado (versionado no Git)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Decisões de design registradas

- **Scripts de uso único fora de `src/app/`**: `prepare_data.py`, `train.py` e a DAG do Airflow nunca são importados pela API — comunicam-se apenas via artefatos em disco (desacoplamento por artefato, não por import direto).
- **`TextPreprocessor` sem `save`/`load` na interface** (diferente de `UrgencyClassifier`): persistência do preprocessor é feita via `joblib.dump`/`joblib.load` diretamente no script de treino, evitando inflar o contrato da interface com uma capacidade genérica já resolvida por ferramenta externa — decisão deliberada, não uma inconsistência.
- **`UrgencyClassifier.classes()`**: adicionado à interface para evitar que a API acessasse `sklearn` diretamente (`classifier.classifier.classes_`), preservando a promessa do Strategy Pattern de que a implementação concreta pode ser trocada sem alterar a API.
- **ONNX só no classificador, não no TF-IDF**: o vetorizador permanece sklearn (esparso). O `OnnxClassifier` densifica float32 só na inferência e devolve label + probabilidade numa única `InferenceSession.run`, no lugar de dois passes sklearn (`predict` + `predict_proba`).


