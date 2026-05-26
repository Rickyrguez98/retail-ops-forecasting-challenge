# DECISIONS — Trade-offs técnicos

Documento de decisiones tomadas durante el challenge. Cada entrada incluye **opciones consideradas**,
**criterio**, y **lo elegido**.

---

## 1. Problema de negocio: demanda + cash (no sólo uno)

**Opciones**: (a) sólo demand forecasting, (b) sólo cash optimization, (c) ambos como sistema integrado.

**Criterio**: el dataset soporta los dos targets con calidad; conectarlos demuestra entendimiento de operaciones retail (la demanda alimenta el cash, y ambos comparten features temporales).

**Elegido**: (c). Demand al nivel store × category × day; cash al nivel store × day.
> Si hubiera sido un proyecto productivo, demand alimentaría reposición y cash alimentaría logística de carga.

---

## 2. Target de demand: `total_transactions` (no `units_sold`)

**Opciones**: `total_transactions`, `units_sold`, `amount_total`.

**Criterio**:
- `total_transactions` tiene 0 nulos vs. 6,118 en `units_sold`.
- Es la unidad operativa para staffing y capacidad de checkout.
- `amount_total` mezcla mix de productos (efecto precio); operativamente menos limpio.

**Elegido**: `total_transactions`.

---

## 3. Granularidad de cash: store × day (no store × category × day)

**Criterio**: el efectivo se maneja a nivel tienda en operaciones reales. La caja general consolida categorías. Forecastar por categoría introduce ruido y desconecta del uso.

**Elegido**: store × day, sumando categorías. Las categorías quedan agregadas; `total_transactions` lagged a nivel store-day entra como feature auxiliar.

---

## 4. Split temporal (no random)
**Periodo**: 2023-01-01 → 2024-02-29 (~14 meses).

**Decisión**:
- **Train**: 2023-01-01 → 2023-09-30 (9 meses, todas las estaciones excepto el segundo Buen Fin).
- **Val**: 2023-10-01 → 2023-11-30 (incluye Buen Fin para que el tuning vea el evento de mayor estrés).
- **Test**: 2023-12-01 → 2024-02-29 (Navidad + post-temporada; condiciones diversas).

**Por qué no random split**: en series temporales, random split filtra futuro al pasado vía vecindad temporal y vía correlaciones cross-entity.

---

## 5. Leakage: `replenishment_signal` excluida + blocklist explícita

**Evidencia**: el diccionario de datos dice *“calculada internamente con base en la demanda observada”*. Verifiqué que `corr(replenishment_signal, total_transactions) ≈ 1.0` en el reporte de calidad.

**Mitigación**:
1. Excluida vía `leakage_blocklist` en `configs/config.yaml`.
2. Test (`test_select_feature_columns_drops_blocklist`) verifica que no entra a las columnas predictoras.
3. Adicionalmente excluí componentes same-day del target (`cash_transactions`, `amount_cash`, `amount_card`, `amount_total`, `card_transactions`, `units_sold`, `avg_ticket`) para no permitir reconstruir el target desde sus partes.

**Lo que sí se permite**:
- `has_promotion`: asumido planeado con anticipación. Documentado; ablation pendiente como mejora futura.
- Calendar flags (`is_payday`, `is_holiday`, etc.): legítimamente conocidos a priori.
- Store metadata: estática.

---

## 6. Modelo: HistGradientBoostingRegressor + LightGBM benchmark

**Opciones**: HistGradientBoosting (sklearn), LightGBM, XGBoost.

**Criterio**:
- **HGB como modelo principal**: sin dependencias externas, sklearn-native, reproducibilidad simple en CI, soporta early stopping.
- **LightGBM como benchmark tuneado**: integrado con **Optuna Bayesian optimization** (TPE sampler) sobre 9 hiperparámetros (`learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `subsample`, `colsample_bytree`, `subsample_freq`).
- 30 trials de Optuna minimizan WAPE en val. La configuración ganadora se refit y se evalúa en test con el mismo pipeline de event uplift que HGB.
- XGBoost descartado: peor manejo nativo de categoricals que LightGBM y no ofrece ventaja diferencial en este tamaño de datos.

**Resultado**: LightGBM tuneado supera a HGB por ~0.2 pp en test (WAPE 0.247 vs 0.249 post-uplift). Ambos modelos están integrados en el pipeline (`make train-demand-lgb`) y trackados separadamente en MLflow.

---

## 7. Métrica primaria: WAPE

**Opciones**: MAPE, sMAPE, WAPE, MASE, RMSE.

**Criterio**:
- **MAPE** explota con ceros (Express stores en categorías de baja venta).
- **sMAPE** simétrico pero menos interpretable para negocio.
- **WAPE** = `sum(|err|) / sum(|y|)` — un número, en unidades del target, robusto a ceros, intuitivo.
- **RMSE** secundaria para detectar outliers grandes.
- **Bias** secundaria para detectar sesgo sistemático (importante en cash: bias > 0 significa que se quedan cortos).

**Elegido**: WAPE primaria; MAE, RMSE, sMAPE, Bias secundarias.

---

## 8. Cash buffer: P90 de residuos por tienda

**Opciones**:
1. Buffer constante (e.g., 20%).
2. Quantile regression (entrenar un segundo modelo de cuantiles).
3. Conformal prediction.
4. **P-quantile de residuos absolutos en validación, por tienda.**

**Criterio**: (4) es interpretable, defendible, y no requiere un segundo modelo. La explicación al negocio es: *“en el 90% de los días pasados, el error fue ≤ X pesos; cargamos esa cantidad extra como colchón”*.

**Elegido**: (4). Q se parametriza en `configs/config.yaml` (default 0.90). Cuando hay tiempo, conformal sería el siguiente paso.

**Limitación**: no optimiza denominaciones (no hay datos de denominaciones, balance de apertura, ni calendario de recolección). Es un proxy de planeación.

---

## 9. Categoricals: encoded integer codes (no one-hot, no target encoding)

**Opciones**: one-hot, label encoding, target encoding, sklearn native.

**Criterio**: HGB en sklearn 1.4+ soporta categoricals nativos pero el entorno tiene 1.2.2. Target encoding introduce leakage si no se hace con cross-validation por fold. Label encoding (`category.cat.codes`) preserva tree splits razonablemente.

**Elegido**: integer codes via `features.encode_categoricals`. Reproducible, sin leakage.

---

## 10. MLflow: file-based local

**Opciones**: file-based, sqlite, remote tracking server.

**Criterio**: el challenge se entrega como repo público; un tracking server remoto no aporta. File-based `./mlruns` es portable y reproducible.

**Elegido**: `file:./mlruns`. Hay un fallback JSON sidelog (`reports/*_tracking_sidelog.json`) si MLflow no es importable en el entorno del reviewer.

---

## 11. Datos faltantes en `amount_cash` (5.9%)

**Opciones**:
1. Imputar (media, mediana, model-based).
2. **Excluir filas con `amount_cash` nulo del entrenamiento Y de la evaluación.**

**Criterio**: las nulls son por “fallas de POS y conectividad”. Imputar ahí inventaría señal. Excluir es honesto y restringe el alcance del modelo a días con datos válidos. El % de exclusión se reporta en `data_quality_report.md`.

**Elegido**: (2). Reporte de cobertura en métricas.

---

## 12. Predicciones clippadas a ≥ 0

Demand y cash son no-negativas por definición. Sin clipping, una predicción negativa rompe la interpretación operativa. Aplicado en `modeling.assemble_predictions` y `cash_forecasting.predict_cash`.
