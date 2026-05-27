# Flujo de Trabajo por Agentes

Este documento describe el patrón de "agentes especializados" usado durante
el desarrollo. Los agentes no son procesos separados — son contratos mentales
que separan responsabilidades. Cada uno tiene entradas, salidas, y un
criterio de “done” explícito.

## Agentes y contratos

### A. Project Architect
- **Entrada**: el challenge prompt + inspección de los CSVs.
- **Salida**: estructura de carpetas, `configs/config.yaml`, leakage blocklist,
  splits temporales acordados.
- **Done**: scaffold compila, `make help` funciona.

### B. Data Quality
- **Entrada**: CSVs en `data/raw/`.
- **Salida**: `reports/data_quality_report.md` + tests en `tests/test_data_validation.py`.
- **Done**: 11 checks pasan, missingness reportada.

### C. EDA / Stats
- **Entrada**: dataset crudo + report de calidad.
- **Salida**: notebooks 01/02, hipótesis confirmadas o descartadas escritas en PROCESS.md.
- **Done**: estacionalidad, eventos y formato/región caracterizados.

### D. Feature Engineering
- **Entrada**: dataset merged + leakage blocklist.
- **Salida**: `src/features.py` + `src/splits.py` + tests anti-leakage.
- **Done**: tests críticos (`test_lag_features_do_not_use_same_day_target`,
  `test_rolling_mean_uses_only_past`) pasan.

### E. Demand Forecasting
- **Entrada**: feature matrix demand + splits.
- **Salida**: baselines + HGB + predicciones en `data/processed/demand_predictions_*.csv`.
- **Done**: HGB supera el mejor baseline en WAPE val.

### F. Cash Forecasting
- **Entrada**: store-day feature matrix.
- **Salida**: HGB + buffers + `reports/cash_forecast_recommendations.csv`.
- **Done**: coverage P90 ≥ 0.80 en validación.

### G. MLflow Tracking
- **Entrada**: configuración + cada run.
- **Salida**: experiments en `./mlruns` + summary JSON.
- **Done**: cada run tiene params + métricas + tag de familia.

### H. Testing & Quality
- **Entrada**: cada módulo en `src/`.
- **Salida**: tests en `tests/` que rompen al introducir un bug obvio.
- **Done**: `make test` pasa en una corrida limpia.

### I. Documentation
- **Entrada**: el repo completo.
- **Salida**: README, PROCESS, AI_USAGE, DECISIONS, model_card, final_report.
- **Done**: un reviewer externo puede correr el proyecto y entender las decisiones sin preguntar.

### J. Review
- **Entrada**: repo completo.
- **Salida**: `reports/review_checklist.md` con verde/rojo/amarillo por sección.
- **Done**: ningún rojo crítico abierto en el entregable.

## Patrón de ejecución

Secuencial con bucles cortos cuando una decisión upstream forzaba revisión.
Por ejemplo, cuando E (demand) detectó que `amount_card` se había colado vía
`amount_total`, regresó a A (architect) para ampliar el blocklist y a D
(features) para añadir el test correspondiente.

## Convención

- Cada agente trabaja sobre el directorio raíz; no hay isolación por carpeta.
- Los commits siguen Conventional Commits y referencian el agente
  responsable cuando aplica (`feat(features): ...` ≈ Feature Engineering agent).
