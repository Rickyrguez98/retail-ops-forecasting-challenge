# Retail Operations Forecasting — Walmart Senior DS Challenge

A Mexican retail chain (80 tiendas × 6 categorías, datos diarios 2023-01-01 → 2024-02-29) necesita
planear **cuántos cajeros tener disponibles** y **cuánto efectivo cargar por sucursal**. Este repositorio resuelve esas
dos decisiones con un mismo backbone de features temporales:

| Tarea | Granularidad | Target | Uso de negocio |
|-------|--------------|--------|----------------|
| **Demanda** | tienda × categoría × día | `total_transactions` | Staffing y capacidad de checkout |
| **Efectivo** | tienda × día | `amount_cash` + buffer P90 | Recomendación operativa de carga de efectivo |

Detalle del razonamiento, trade-offs y resultados → [`reports/final_report.md`](reports/final_report.md).

---

## Cómo correr

```bash
# 1. Instalar dependencias (sklearn 1.2.2 + mlflow + pytest)
make install

# 2. Verificación rápida (tests deterministas, no requieren los CSVs)
make test

# 3. Pipeline end-to-end (≈ 5–10 min en una laptop estándar)
make all   # quality -> prepare -> train-demand -> train-cash -> evaluate -> report
```

Targets individuales (idempotentes):

```bash
make quality        # data quality report -> reports/data_quality_report.md
make prepare        # interim + processed parquet/csv
make train-demand   # baselines + HGB, MLflow tracking
make train-cash     # HGB + per-store P90 buffer
make evaluate       # segment/event breakdowns + figures
make report         # collates everything into reports/experiment_summary.md
```

## Estructura

```
configs/        YAML config (paths, splits, leakage blocklist)
data/raw/       Inmutable (transactions, stores, calendar, diccionario)
data/interim/   Joined + aggregated (regenerable)
data/processed/ Feature matrices (regenerable)
notebooks/      EDA + análisis (importan desde src/)
src/retail_ops_forecasting/
                config, data, validation, features, splits, baselines,
                modeling, cash_forecasting, metrics, tracking, reporting, utils
scripts/        CLI entry points (orquestación, no lógica)
tests/          pytest — schemas, splits, metrics, anti-leakage
reports/        Métricas, figuras, model card, final report
```

## Decisiones clave

- **Split temporal**: train 2023-01 → 2023-09, val 2023-10 → 2023-11 (incluye Buen Fin para tuning),
  test 2023-12 → 2024-02 (Navidad + post-temporada). Nunca random split.
- **Leakage**: `replenishment_signal` excluida (el diccionario indica que se calcula con base en la demanda observada).
  La lista completa de columnas vetadas está en [`configs/config.yaml`](configs/config.yaml).
- **Métrica primaria**: WAPE — robusto a ceros, interpretable, alineado con planeación retail.
- **Cash recommendation**: regla transparente `recommended = clip(pred, 0) + buffer_P90(store)`.

Detalle por decisión → [`DECISIONS.md`](DECISIONS.md).

## Documentación

- [`PROCESS.md`](PROCESS.md) — diario de proceso por fase, herramientas usadas, qué funcionó y qué no
- [`AI_USAGE.md`](AI_USAGE.md) — transparencia sobre uso de Claude (qué se generó vs. qué se validó manualmente)
- [`DECISIONS.md`](DECISIONS.md) — trade-offs técnicos y de modelado
- [`reports/model_card.md`](reports/model_card.md) — model card estilo Google
- [`reports/final_report.md`](reports/final_report.md) — resultados, recomendaciones, limitaciones
- [`reports/experiment_summary.md`](reports/experiment_summary.md) — generado por `make report`

## Reproducibilidad

- Semilla fija (`seed: 42` en `configs/config.yaml`).
- Versiones pinneadas en `requirements.txt`.
- MLflow tracking local en `./mlruns` (fallback JSON si MLflow no está disponible).
- Tests deterministas verifican splits, métricas y anti-leakage.
