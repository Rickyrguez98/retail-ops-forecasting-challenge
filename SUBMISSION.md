# Resumen de la Entrega — Walmart Senior DS Challenge

Guía rápida para el revisor. Toda la profundidad técnica está en los documentos
listados abajo; este archivo es solo un mapa.

---

## Problema de negocio

Una cadena retail mexicana (80 tiendas × 6 categorías × 425 días) necesita planear
dos cosas, todos los días:

1. **Pronóstico de demanda** por tienda × categoría × día → alimenta planeación de
   staffing y capacidad de checkout.
2. **Pronóstico de requerimiento de efectivo** por tienda × día → alimenta logística
   de carga y minimiza tanto el riesgo de quedarse corto como el costo del cash
   inmovilizado.

Ambas decisiones comparten los mismos patrones (estacionalidad semanal, eventos,
quincenas, características de tienda y entorno socioeconómico), por lo que las modelo
como un sistema con un backbone común y dos heads.

---

## Qué leer primero (en este orden)

| # | Archivo | Por qué |
|---|---------|---------|
| 1 | [`README.md`](README.md) | Instalación, comandos, resumen del proyecto |
| 2 | [`PROCESS.md`](PROCESS.md) | Cronología y razonamiento por fase del trabajo |
| 3 | [`reports/final_report.md`](reports/final_report.md) | **Resultados** + selección de modelo + significado del WAPE + separación clean holdout vs. escenario post-hoc |
| 4 | [`reports/model_card.md`](reports/model_card.md) | Detalles técnicos del modelo, métricas por segmento, features, controles de leakage |
| 5 | [`DECISIONS.md`](DECISIONS.md) | Decisiones técnicas con alternativas consideradas y criterio (split, target, leakage, modelo, capa post-hoc, métrica, buffer) |
| 6 | [`AI_USAGE.md`](AI_USAGE.md) | Transparencia sobre qué generé yo y qué generó el LLM |

Para revisar código rápido:
- [`src/retail_ops_forecasting/features.py`](src/retail_ops_forecasting/features.py) — lags y rolling con `shift(>=1)` (anti-leakage)
- [`tests/test_features.py`](tests/test_features.py) — tests que validan que el lag no toca same-day
- [`configs/config.yaml`](configs/config.yaml) — splits, leakage blocklist, capa post-hoc documentada como regla de negocio
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — pipeline de CI (black + isort + ruff + pytest)

---

## Fortalezas principales

- **Validación temporal explícita**: train 2023-01 → 09 / val 2023-10 → 11 (incluye
  Buen Fin para tuning) / test 2023-12 → 2024-02 (held-out, incluye Navidad).
  Verificado por test (`test_splits.py::test_split_partitions_are_disjoint_and_ordered`).
- **Prevención de leakage** auditable: `leakage_blocklist` en `configs/config.yaml`,
  todos los lags y rolling con `shift(>=1)`, 3 tests críticos que fallan si se
  rompe el invariante (`test_features.py`).
- **Reporte limpio: clean holdout separado del ajuste post-hoc**. El desempeño
  oficial out-of-sample es WAPE = 0.293 (LightGBM tuneado, sin ajustes). La capa
  *event stress adjustment* se reporta en una tabla aparte como escenario operativo
  y NO como performance del modelo. Razonamiento en
  [`final_report.md §4.1`](reports/final_report.md) y
  [`DECISIONS.md §7`](DECISIONS.md).
- **Modelo elegido con justificación cuantitativa**: LightGBM tuneado con Optuna
  Bayesian (TPE, 30 trials, 9 hiperparámetros) gana en WAPE / MAE / RMSE de forma
  consistente. HGB queda como fallback. Comparativa completa en
  [`final_report.md §4.3`](reports/final_report.md).
- **MLflow integrado** con Model Registry (3 modelos versionados con signature),
  Dataset lineage (9 datasets trackeados), y artifacts (predicciones + figuras +
  breakdowns) por run.
- **Docker**: imagen multi-stage + `docker-compose.yml` con servicios `pipeline`,
  `tests` y `mlflow` (UI en host port 5001 para evitar AirPlay en macOS).
- **CI con GitHub Actions**: lint (black + isort + ruff) y tests en cada push/PR.
- **Regla de cash operativa y auditable**: `recommended = max(0, ŷ) + P90_residuos_val(store)`,
  un solo modelo + un buffer transparente, coverage test = 94.6 % (target ≥ 90 %).
- **Tests deterministas** (19/19): schemas, splits, métricas, anti-leakage, cash buffer.
- **Transparencia sobre el uso de AI**: [`AI_USAGE.md`](AI_USAGE.md) lista qué
  prompts usé, qué outputs del LLM rechacé y por qué.

---

## Limitaciones conocidas

- **Sin granularidad por denominación de billetes**: el dataset no tiene cash
  drawer por denominación; el sistema recomienda monto total, no la composición.
- **Sin saldos de apertura/cierre por tienda**: no se modela el inventario de
  efectivo entre días, solo el flujo proyectado del día.
- **Sin calendario de recolecciones**: no se sabe cuándo cash retira efectivo de
  la tienda, lo que afecta la optimización de carga (treated as exogenous).
- **Capa post-hoc requiere re-calibración antes de producción**: los multiplicadores
  Dic 24-25 / Dic 31 son defaults basados en estacionalidad publicada del retail
  mexicano, no validados con múltiples años de la propia cadena.
- **Pronóstico diario, no intra-día**: el sistema no captura el patrón horario
  dentro del día (apertura → pico tarde → cierre). Para staffing por turno se
  necesitaría agregación horaria.
- **Horizonte recomendado: 1-7 días**. No usar el modelo para horizontes mayores
  sin re-entrenamiento explícito (los lags y rollings dejan de tener sentido).
- **Sin manejo de tiendas / categorías nuevas** (cold start): el modelo necesita
  historial para los lags. Una tienda nueva caería al fallback baseline.

---

## Cómo reproducir

Hay tres rutas. Todas regeneran los mismos artefactos.

### Ruta A — Local con virtualenv

```bash
git clone https://github.com/Rickyrguez98/retail-ops-forecasting-challenge.git
cd retail-ops-forecasting-challenge

python3.11 -m venv .venv && source .venv/bin/activate
make install            # pip install -r requirements.txt
make test               # 19 tests en ~1 s
make lint               # black + isort + ruff (mismo que CI)
make all                # quality → prepare → train-demand → train-demand-lgb → train-cash → evaluate → report
```

Comandos individuales del Makefile:
- `make quality` — checks de calidad de datos sobre los CSV raw
- `make prepare` — features y splits → `data/processed/*.parquet`
- `make train-demand` — baselines + HGB + capa post-hoc opcional
- `make train-demand-lgb` — LightGBM tuneado con Optuna (`N_TRIALS=30` por defecto)
- `make train-cash` — modelo de cash + buffer P90 por tienda
- `make evaluate` — métricas finales en test + breakdowns por segmento
- `make report` — `experiment_summary.md` consolidado
- `make format` — auto-format con black/isort/ruff --fix

### Ruta B — Docker

```bash
git clone https://github.com/Rickyrguez98/retail-ops-forecasting-challenge.git
cd retail-ops-forecasting-challenge

docker compose build
docker compose run --rm tests           # 19/19 passed
docker compose run --rm pipeline        # equivalente a `make all` dentro del container
docker compose up mlflow                # MLflow UI en http://localhost:5001
```

### Ruta C — CI

Cada push a `main` o pull request dispara [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
que ejecuta los mismos `make lint` y `make test` que se corren localmente.

---

## Métricas headline

| Modelo | Métrica | Test (out-of-sample) | Notas |
|--------|---------|---------------------:|-------|
| LightGBM tuneado (oficial) | Demand WAPE | **0.293** | −9.4 % rel. vs. baseline naive lag-7 (0.323) |
| HGB (fallback) | Demand WAPE | 0.296 | Mismo pipeline; backup operativo |
| HGB | Cash WAPE | 0.153 | Buffer P90 cubre el sub-forecast |
| HGB + buffer P90 | Cash coverage | **94.6 %** | Target ≥ 90 % |

> El escenario operativo con la capa post-hoc llevaría el WAPE de demanda a 0.247,
> pero ese número se reporta como análisis de escenario, no como performance
> out-of-sample del modelo. Detalle en
> [`final_report.md §4.1.B`](reports/final_report.md).
