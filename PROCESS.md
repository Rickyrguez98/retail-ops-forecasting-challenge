# PROCESS — Diario de trabajo

Este documento describe el **cómo**: orden de fases, decisiones tomadas en vivo,
herramientas usadas. La narrativa de **qué se
construyó y por qué** vive en [`reports/final_report.md`](reports/final_report.md);
los trade-offs técnicos detallados están en [`DECISIONS.md`](DECISIONS.md);
el uso de IA está en [`AI_USAGE.md`](AI_USAGE.md).

## Estructura del trabajo

Adopté un patrón de **agentes especializados** documentado en `.claude/AGENT_WORKFLOW.md`:

| Agente | Fase | Responsabilidad |
|--------|------|-----------------|
| Project Architect | 0 | Definir estructura, configs, contrato de leakage |
| Data Quality | 1 | Loaders, schema validation, reporte de calidad |
| EDA / Stats | 2 | Notebooks de exploración + análisis estadístico |
| Feature Engineering | 3 | Lags/rolling con `shift(>=1)`, tests anti-leakage |
| Demand Forecasting | 4 | Baselines + HGB + tracking |
| Cash Forecasting | 5 | HGB store-day + P90 buffer rule |
| MLflow Tracking | 4–5 | Logging de params/métricas/artefactos |
| Testing & Quality | transversal | pytest, ruff, black, isort, pre-commit |
| Documentation | 7 | README, PROCESS, AI_USAGE, DECISIONS, model card, final report |
| Review | 8 | Checklist senior, reproducibilidad |

Los agentes son una orquestación mental — no procesos separados. Cada uno tenía
un “contrato” explícito de entradas/salidas; lo aplicaba secuencialmente con
bucles cortos cuando una decisión upstream forzaba revisión downstream.

## Fases ejecutadas

### Fase 0 — Setup
- Estructura de carpetas, `pyproject.toml`, `requirements.txt` pinneados.
- `configs/config.yaml` con splits temporales y **leakage blocklist** explícita.
- `Makefile` con targets idempotentes y `.pre-commit-config.yaml` para black/isort/ruff.
- Tests iniciales sobre datos sintéticos (fixtures en `tests/conftest.py`).

### Fase 1 — Calidad de datos
- Loaders por archivo (`data.py`) que normalizan dtypes y parsean fechas.
- `validation.py` corre 11 checks: esquemas, claves únicas, continuidad de calendario,
  rangos de valores, reconciliación de cash+card vs total, y flag de leakage en
  `replenishment_signal` (correlación con same-day demand, documentada).
- Reporte generado en `reports/data_quality_report.md`.

### Fase 2 — EDA y análisis estadístico
- `notebooks/01_initial_eda.ipynb`: shape, missingness, distribución temporal,
  patrones por tienda/categoría/región.
- `notebooks/02_statistical_analysis.ipynb`: ANOVA por formato/región/categoría,
  efecto de eventos (Buen Fin, Semana Santa, Navidad, quincenas), estacionalidad
  intra-semanal y mensual.
- Hipótesis confirmadas:
  - Estacionalidad semanal clara (fines de semana ↑).
  - Buen Fin y temporada navideña elevan demanda en categorías no-alimentarias.
  - `replenishment_signal` correlaciona ~1.0 con `total_transactions` same-day → leakage confirmado.

### Fase 3 — Features y splits
- `features.py`:
  - Calendar interactions (`dow_sin/cos`, `month_sin/cos`, `is_payday × is_weekend`, `store_age_years`).
  - Lags 1/7/14/28 y rolling mean/std con `shift(>=1)` agrupado por entidad.
  - `trend_7v28` como proxy de tendencia corta vs. media.
- `splits.py`: separado de features para que el código de validación temporal sea inspeccionable.
- Tests anti-leakage críticos en `tests/test_features.py`.

### Fase 4 — Demand forecasting
- Baselines registrados con MLflow:
  1. Seasonal naive (lag-7) → WAPE val 0.419 / test 0.323
  2. Rolling mean 28 → WAPE val 0.348 / test 0.402
  3. Historical day-of-week mean por (store, category) → WAPE val 0.313 / test 0.335
- Modelo: `HistGradientBoostingRegressor` con categoricals encodeados a códigos enteros (sklearn 1.2.2 no soporta strings nativos).
- **Resultado HGB v1**: WAPE val 0.262 / test 0.296 (mejor que el mejor baseline por 2.7-5.1 pp).
- Predicciones clippadas a ≥0 (demanda no-negativa).
- Resultados → `reports/demand_experiment_summary.json` + `reports/demand_overall_test_metrics.csv`.

### Fase 5 — Cash forecasting
- Granularidad: tienda × día (suma de categorías; el efectivo se maneja a nivel tienda en operaciones).
- Features de cash propias: `cash_lag_{1,7,14,28}`, rolling mean/std, `cash_share_lag_1` (cash/total ayer).
- Buffer rule: `recommended = max(0, ŷ) + P90(|residuo_val|)` por tienda.
- **Resultado**: WAPE val 0.170 / test 0.153; coverage P90 val 0.942 / test 0.946 (supera el target de 0.90).
- Métrica adicional: `coverage` (fracción de días donde `recommended ≥ actual`).

### Fase 6 — Error analysis
- Breakdowns por categoría, región, formato y `event_label` (Buen Fin / Semana Santa / Navidad / quincena / weekend / regular).
- Notebook 04 con análisis cualitativo de residuos y figuras (actual vs pred, hist residuos).

### Fase 7 — Documentación
- README compacto + apunta a docs largos.
- `DECISIONS.md` con razonamiento por decisión.
- `AI_USAGE.md` con transparencia.
- `reports/model_card.md` estilo Google.
- `reports/final_report.md` con resultados y recomendaciones de negocio.

### Fase 8 — Review
- Checklist final en `reports/review_checklist.md`.
- Reproducibilidad verificada vía `make clean && make all`.

## Herramientas usadas

- **Python 3.11**, pandas 1.5.3, scikit-learn 1.2.2, scipy 1.10.1
- **MLflow** local file-store en `./mlruns` (con fallback JSON sidelog si MLflow no es importable)
- **pytest** para tests deterministas
- **Black / isort / ruff** vía pre-commit
- **Claude Code** (Anthropic) como asistente de generación de scaffolding y revisión.
  Detalle de uso, qué se generó vs. qué se validó manualmente → [`AI_USAGE.md`](AI_USAGE.md).


## Convenciones de commits

Conventional commits. Un commit = una responsabilidad. Los commits `exp(...)` registran resultados
de experimentos en el mensaje. Ver `git log --oneline` para el historial completo.
