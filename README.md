# Retail Operations Forecasting — Walmart Senior DS Challenge

A Mexican retail chain (80 tiendas × 6 categorías, datos diarios 2023-01-01 → 2024-02-29) necesita
planear **cuántos cajeros tener disponibles** y **cuánto efectivo cargar por sucursal**. Este repositorio resuelve esas
dos decisiones con un mismo backbone de features temporales:

| Tarea | Granularidad | Target | Uso de negocio |
|-------|--------------|--------|----------------|
| **Demanda** | tienda × categoría × día | `total_transactions` | Staffing y capacidad de checkout |
| **Efectivo** | tienda × día | `amount_cash` + buffer P90 | Recomendación operativa de carga de efectivo |

**Resultados oficiales (test holdout limpio 2023-12 → 2024-02, sin ajustes post-hoc):**
demanda **WAPE = 0.293** (LightGBM tuneado, modelo elegido) / **0.296** (HGB, fallback) ·
mejor baseline 0.323 → **−9.4 % WAPE relativo** ·
efectivo **WAPE = 0.153** · coverage P90 = **94.6 %**.

Adicionalmente reporto un escenario operativo con una capa post-hoc de *event stress
adjustment* (regla de negocio sobre picos navideños, **no calibrada contra test**)
que llevaría el WAPE de demanda a 0.247. Detalle metodológico y separación clean
holdout vs. escenario post-hoc → [`reports/final_report.md` §4.1.A / §4.1.B](reports/final_report.md).

---

## Contenido

- [Requisitos](#requisitos)
- [Opción A — Entorno virtual (local)](#opción-a--entorno-virtual-local)
- [Opción B — Docker](#opción-b--docker)
- [Targets del Makefile](#targets-del-makefile)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Decisiones clave](#decisiones-clave)
- [Documentación](#documentación)

---

## Requisitos

- **Python 3.11** (local) o **Docker Desktop** (contenedor)
- Git

---

## Opción A — Entorno virtual (local)

### 1. Clonar el repositorio

```bash
git clone https://github.com/Rickyrguez98/retail-ops-forecasting-challenge.git
cd retail-ops-forecasting-challenge
```

### 2. Crear y activar el entorno virtual

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows (PowerShell)
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .               # instala el paquete src/ en modo editable
```

### 4. Verificar instalación (tests)

```bash
make test       # 19 tests deterministas, ~1 s
```

### 5. Correr el pipeline completo

```bash
make all        # quality → prepare → train-demand → train-cash → evaluate → report
```

Los artefactos se generan en `data/interim/`, `data/processed/`, `models/`, `reports/` y `mlruns/`.

### 6. Ver experimentos en MLflow UI (opcional)

```bash
mlflow ui --backend-store-uri ./mlruns
# Abre http://localhost:5000
```

---

## Opción B — Docker

### Requisitos previos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo

### 1. Clonar el repositorio

```bash
git clone https://github.com/Rickyrguez98/retail-ops-forecasting-challenge.git
cd retail-ops-forecasting-challenge
```

### 2. Construir la imagen

```bash
docker compose build
# o sin compose:
docker build -t retail-ops-forecasting:latest .
```

### 3. Correr el pipeline completo

```bash
docker compose run --rm pipeline
```

Los artefactos se montan en los directorios locales (`models/`, `reports/`, `mlruns/`, `data/`) via volúmenes, por lo que persisten en tu máquina al terminar el contenedor.

### 4. Correr solo los tests

```bash
docker compose run --rm tests
```

### 5. Ver MLflow UI en Docker

```bash
docker compose up mlflow
# Abre http://localhost:5001
```

**Qué verás en el UI:**
- **Experiments**: `demand_forecasting` (baselines + HGB + LightGBM), `cash_forecasting` (HGB + buffer), `evaluation` (figuras + breakdowns). Cada run con params, metrics, tags.
- **Models** (Model Registry): `demand_hgb`, `demand_lgb`, `cash_hgb` — cada uno con signature de entrada/salida e input example.
- **Datasets**: 9 datasets registrados (`demand_train`/`val`/`test` y `cash_train`/`val`/`test`) con source, schema y row count.
- **Artifacts por run**: model files, predicciones CSV, breakdowns por segmento, figuras de actual-vs-pred y residuos.

> **Nota sobre el puerto en macOS**: el puerto **host por defecto es 5001** (no 5000) porque macOS Monterey+ tiene **AirPlay Receiver** escuchando en el 5000 y lo bloquea. Si prefieres usar otro puerto: `MLFLOW_HOST_PORT=5050 docker compose up mlflow`.

> **Nota:** la primera vez que construyes la imagen tarda ~3-5 min mientras descarga dependencias. Las ejecuciones siguientes usan el layer cache de Docker y son instantáneas.
>
> **Espacio en disco requerido:** la imagen final pesa ~1.5 GB (Python 3.11-slim + pandas + scikit-learn + lightgbm + mlflow). Asegúrate de tener al menos **3 GB libres** en el disco anfitrión antes de construir.

---

## Targets del Makefile

```bash
make install            # pip install -r requirements.txt
make test               # 19 pytest deterministas (no requieren CSVs)
make quality            # data quality report → reports/data_quality_report.md
make prepare            # parquets interim + processed
make train-demand       # baselines + HGB + event-window uplift, MLflow tracking
make train-demand-lgb   # LightGBM benchmark + Optuna Bayesian tuning (N_TRIALS=30 default)
make train-cash         # HGB + per-store P90 buffer
make evaluate           # breakdowns por segmento/evento + figuras
make report             # consolida todo → reports/experiment_summary.md
make all                # quality → prepare → train-demand → train-demand-lgb → train-cash → evaluate → report
make clean              # elimina artefactos generados (mantiene raw data y reports)
```

> **Personalizar el tuning de LightGBM**: `make train-demand-lgb N_TRIALS=50` corre 50 trials de Optuna en lugar de 30.

---

## Estructura del proyecto

```
.
├── configs/                    YAML: splits, leakage blocklist, model params, logging
├── data/
│   ├── raw/                    Datos originales (inmutables)
│   ├── interim/                Tablas joined + aggregated (regenerables con make prepare)
│   └── processed/              Feature matrices (regenerables)
├── notebooks/                  EDA + análisis estadístico + experimentos (importan src/)
│   ├── 01_initial_eda.ipynb
│   ├── 02_statistical_analysis.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_error_analysis.ipynb
├── src/retail_ops_forecasting/ Paquete instalable
│   ├── config.py               Typed YAML config loader (frozen dataclasses)
│   ├── data.py                 Loaders + build_merged + cash_store_day
│   ├── validation.py           11 data quality checks
│   ├── features.py             Lags/rolling con shift≥1 + calendar + store metadata
│   ├── splits.py               time_based_split + walk_forward_folds
│   ├── baselines.py            Seasonal naive, rolling mean, historical DOW mean
│   ├── modeling.py             HistGradientBoosting wrapper
│   ├── cash_forecasting.py     Cash HGB + per-store P90 buffer rule
│   ├── metrics.py              WAPE, MAE, RMSE, sMAPE, bias, coverage
│   ├── tracking.py             MLflow wrapper + JSON sidelog fallback
│   ├── reporting.py            Experiment summary generator
│   └── utils.py                set_seed + setup_logging
├── scripts/                    CLI entry points (orquestación, sin lógica de negocio)
├── tests/                      pytest — schemas, splits, métricas, anti-leakage (19 tests)
├── reports/                    Métricas, figuras, model card, final report
├── Dockerfile                  Multi-stage build (base → deps → app)
├── docker-compose.yml          pipeline / tests / mlflow services
├── Makefile                    Targets idempotentes
├── requirements.txt            Versiones pinneadas
└── pyproject.toml              black / isort / ruff / pytest config
```

---

## Decisiones clave

- **Split temporal**: train 2023-01 → 2023-09, val 2023-10 → 2023-11 (incluye Buen Fin para tuning),
  test 2023-12 → 2024-02 (Navidad + post-temporada). Nunca random split.
- **Leakage**: `replenishment_signal` excluida (el diccionario indica que se calcula con base en la demanda observada).
  Lista completa de columnas vetadas en [`configs/config.yaml`](configs/config.yaml).
- **Métrica primaria**: WAPE — robusto a ceros, interpretable, alineado con planeación retail.
- **Cash recommendation**: regla transparente `recommended = clip(pred, 0) + buffer_P90(store)`.

Detalle por decisión → [`DECISIONS.md`](DECISIONS.md).

---

## Documentación

| Archivo | Contenido |
|---------|-----------|
| [`PROCESS.md`](PROCESS.md) | Diario de proceso por fase, herramientas usadas |
| [`DECISIONS.md`](DECISIONS.md) | 12 trade-offs técnicos y de modelado con alternativas |
| [`AI_USAGE.md`](AI_USAGE.md) | Transparencia sobre uso de IA (qué generó vs. qué se validó manualmente) |
| [`reports/model_card.md`](reports/model_card.md) | Model card estilo Google: métricas, features, componentes implementados, equidad |
| [`reports/final_report.md`](reports/final_report.md) | Resultados, recomendaciones de negocio, componentes del sistema |
| [`reports/experiment_summary.md`](reports/experiment_summary.md) | Generado por `make report` — todas las runs con métricas |

---

## Reproducibilidad

- Semilla fija (`seed: 42` en `configs/config.yaml`).
- Versiones pinneadas en `requirements.txt`.
- MLflow tracking local en `./mlruns` (fallback JSON si MLflow no está disponible).
- 19 tests deterministas verifican splits, métricas y anti-leakage.
- `make clean && make all` reproduce todos los artefactos desde cero.
