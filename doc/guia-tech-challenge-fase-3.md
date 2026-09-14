# MLET — Tech Challenge Fase 3 (spec compacta)

Fonte: `MLET - Tech Challenge Fase 3.pdf` (7 págs.; págs. 1 e 7 = capa). Extraído 2026-09-14.

## Meta

Grupo. Obrigatório. **90% da nota** de todas as disciplinas da fase. Prazo de entrega vale.

**Tema:** Deploy de modelo em produção com CI/CD, monitoramento e otimização de latência.

**Contexto:** Hospital precisa triar laudos (texto) em urgência (`normal` / `atenção` / `urgente`). Classificador NLP **leve**, API REST em Docker. Foco no ciclo de vida: CI/CD (GitHub Actions), retreino orquestrado (Airflow), monitoramento (Prometheus + Grafana), latência.

## Obrigatório (repo)

- CI/CD GitHub Actions: lint → test → build
- Script/DAG Airflow de treino/retreino
- Dockerfile do serviço de inferência
- Stack local: API + Prometheus + Grafana via Docker Compose
- Commits semânticos e organizados

## Vídeo ≤ 5 min (STAR)

- **S:** problema clínico + triagem rápida
- **T:** requisitos (latência, CI/CD, monitoramento)
- **A:** arquitetura, otimização do modelo, monitoração
- **R:** pipeline funcionando, latência alcançada, lições

## Libs exigidas

- sklearn (ou similar) — texto (ex.: TF-IDF + RF ou leve equivalente)
- FastAPI
- prometheus-client
- Airflow

## Boas práticas obrigatórias

- CI/CD com **≥ 2** automações (lint + testes)
- DAG Airflow funcional (dados → treino → salvar modelo)
- Grafana **≥ 3** painéis (ex.: total reqs, latência, taxa de erro)
- **≥ 1** técnica de performance da aula: ONNX, quantização básica **ou** pruning

## Etapas

### 1 — Arquitetura + API (Deploy em Nuvem)

- Decidir nuvem (AWS/Azure/GCP) **batch vs real-time**; documentar no README
- FastAPI: texto do laudo → classificação
- Docker + medir latência **baseline** local
- **Entrega:** API no Docker + texto de decisão arquitetural no README

### 2 — CI/CD + pipeline (CI/CD e Pipeline de Treino)

- GitHub Actions no push: pytest + lint
- DAG Airflow: task lê CSV + task treina e salva modelo
- **Entrega:** workflow YAML + `.py` da DAG

### 3 — Monitoramento (Monitoração de Performance)

- `prometheus_client`: tempo de requisição + contagem
- `docker-compose.yml`: API + Prometheus + Grafana
- Dashboard Grafana das métricas
- **Entrega:** Compose da stack + print/JSON do dashboard

### 4 — Latência + entrega (Latência em Modelos Não Estruturados) ← PENDENTE

- Treinar o classificador de texto
- Aplicar otimização de latência (ex.: **exportar para ONNX Runtime**, ou quantização)
- **Comparar latência original vs otimizado**
- Gravar vídeo STAR
- **Entrega:** modelo otimizado + resultados comparativos de latência + **link do vídeo**

## Rubrica

| Critério | Peso | O que cobram |
|---|---|---|
| Modelagem e otimização | 20% | NLP funcional, conversão/otimização (ex. ONNX) ok, **melhoria de latência demonstrada** |
| CI/CD (Actions) | 15% | workflow + testes básicos |
| Orquestração (Airflow) | 15% | DAG ingestão + treino |
| Monitoramento | 20% | Compose API+Prom+Grafana + dashboard |
| README | 15% | arquitetura nuvem + como executar |
| Vídeo STAR | 15% | demo técnica + impacto, ≤ 5 min |

## Dataset sugerido

Textos médicos / triagem, **≥ 2000** amostras, coluna texto + target.

Exemplos: Medical Abstracts TC Corpus (Kaggle), recortes MIMIC-III (open), ou tabular equivalente.

## Checklist resumido

1. Arquitetura teórica + FastAPI Docker + latência base
2. Actions lint/test + DAG Airflow
3. Compose Prom+Grafana + gerar reqs para gráfico
4. Otimizar (ex. ONNX) + documentar números + vídeo
