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

## 6. Modelo: LightGBM tuneado (oficial) + HistGradientBoosting (fallback)

**Opciones**: HistGradientBoosting (sklearn), LightGBM, XGBoost.

**Criterio**:
- **LightGBM tuneado como modelo oficial**: integrado con **Optuna Bayesian optimization** (TPE sampler) sobre 9 hiperparámetros (`learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `subsample`, `colsample_bytree`, `subsample_freq`). 30 trials de Optuna minimizan WAPE en **val** (no en test). La configuración ganadora se re-entrena y se evalúa **una sola vez** en test.
- **HGB como fallback**: sin dependencias externas, sklearn-native, reproducibilidad trivial en CI, soporta early stopping. Mantiene exactamente el mismo pipeline de features y validación. Permite operar el sistema si LightGBM falla por infraestructura.
- **XGBoost descartado**: peor manejo nativo de categoricals que LightGBM y no ofrece ventaja diferencial en este tamaño de datos.

**Resultado oficial (clean holdout, sin ajustes post-hoc)**: LightGBM tuneado **WAPE test = 0.293**, HGB base **WAPE test = 0.296**. La diferencia es chica pero consistente en todos los segmentos (categoría, formato, evento). Ambos modelos están registrados en MLflow Model Registry (`demand_lgb` y `demand_hgb`, v1 cada uno).

---

## 7. Capa post-hoc: *event stress adjustment* como regla de negocio (no como modelo)

**Decisión**: separar la capa de ajuste por eventos del modelo "oficial" y reportarla
como **escenario operativo** en una tabla independiente, **nunca** como parte del
desempeño out-of-sample del modelo.

**Por qué existe esta capa**: el set de entrenamiento (2023-01 → 2023-09) contiene
exactamente UNA ocurrencia de cada pico retail mexicano (Buen Fin, Navidad, Año Nuevo).
Con n=1 histórico, ningún modelo de árbol puede extrapolar la magnitud del pico de
manera confiable y sub-pronostica sistemáticamente esos días. La industria retail
mexicana sabe **ex-ante** que la demanda y el cash se multiplican ~1.5×–3× en esas
fechas (sector commentary de ANTAD/INEGI). La capa codifica ese conocimiento de
negocio como regla, **no como aprendizaje del modelo sobre el test**.

**Política de calibración**:

| Evento | Cómo se setean los factores | ¿Válido para producción tal cual? |
|--------|-----------------------------|-----------------------------------|
| Buen Fin | Aprendido de **residuos de validación** (val incluye Buen Fin por diseño) | ✅ Legítimo (val-tuned) |
| Dic 24-25 | **Default de regla de negocio** basado en estacionalidad publicada | ⚠️ Requiere re-calibración con 2-3 años previos antes de prod |
| Dic 31 | **Default de regla de negocio** basado en estacionalidad publicada | ⚠️ Requiere re-calibración con 2-3 años previos antes de prod |

**Por qué se reporta separado del modelo "oficial"**: si los factores Dic 24-25 / Dic 31
se ajustaran contra los residuos del test set, sería **leakage metodológico** — el test
holdout dejaría de ser holdout. Para evitar esa ambigüedad la capa se reporta como
escenario operativo (tabla §4.1.B en `reports/final_report.md`) y nunca como desempeño
out-of-sample del modelo.

**Política operativa recomendada**:
1. Producir inicialmente con `event_uplift.enabled: false` (modelo base limpio).
2. Activar la capa solo después de re-calibrar los factores Dic 24-25 / Dic 31 con
   datos históricos de **múltiples años** de la propia cadena.
3. Mantener un A/B contra el modelo base para detectar si la capa empeora algún
   segmento (formato, región, categoría) no presente en los datos de calibración.

---

## 8. Métrica primaria: WAPE

**Opciones**: MAPE, sMAPE, WAPE, MASE, RMSE.

**Criterio**:
- **MAPE** explota con ceros (Express stores en categorías de baja venta).
- **sMAPE** simétrico pero menos interpretable para negocio.
- **WAPE** = `sum(|err|) / sum(|y|)` — un número, en unidades del target, robusto a ceros, intuitivo.
- **RMSE** secundaria para detectar outliers grandes.
- **Bias** secundaria para detectar sesgo sistemático (importante en cash: bias > 0 significa que se quedan cortos).

**Elegido**: WAPE primaria; MAE, RMSE, sMAPE, Bias secundarias.

---

## 9. Cash buffer: P90 de residuos por tienda

**Opciones**:
1. Buffer constante (e.g., 20%).
2. Quantile regression (entrenar un segundo modelo de cuantiles).
3. Conformal prediction.
4. **P-quantile de residuos absolutos en validación, por tienda.**

**Criterio**: (4) es interpretable, defendible, y no requiere un segundo modelo. La explicación al negocio es: *“en el 90% de los días pasados, el error fue ≤ X pesos; cargamos esa cantidad extra como colchón”*.

**Elegido**: (4). Q se parametriza en `configs/config.yaml` (default 0.90). Cuando hay tiempo, conformal sería el siguiente paso.

**Limitación**: no optimiza denominaciones (no hay datos de denominaciones, balance de apertura, ni calendario de recolección). Es un proxy de planeación.

---

## 10. Categoricals: encoded integer codes (no one-hot, no target encoding)

**Opciones**: one-hot, label encoding, target encoding, sklearn native.

**Criterio**: HGB en sklearn 1.4+ soporta categoricals nativos pero el entorno tiene 1.2.2. Target encoding introduce leakage si no se hace con cross-validation por fold. Label encoding (`category.cat.codes`) preserva tree splits razonablemente.

**Elegido**: integer codes via `features.encode_categoricals`. Reproducible, sin leakage.

---

## 11. MLflow: file-based local

**Opciones**: file-based, sqlite, remote tracking server.

**Criterio**: el challenge se entrega como repo público; un tracking server remoto no aporta. File-based `./mlruns` es portable y reproducible.

**Elegido**: `file:./mlruns`. Hay un fallback JSON sidelog (`reports/*_tracking_sidelog.json`) si MLflow no es importable en el entorno del reviewer.

---

## 12. Datos faltantes en `amount_cash` (5.9 %)

**Opciones**:
1. Imputar (media, mediana, model-based).
2. **Excluir filas con `amount_cash` nulo del entrenamiento Y de la evaluación.**

**Criterio**: las nulls son por “fallas de POS y conectividad”. Imputar ahí inventaría señal. Excluir es honesto y restringe el alcance del modelo a días con datos válidos. El % de exclusión se reporta en `data_quality_report.md`.

**Elegido**: (2). Reporte de cobertura en métricas.

---

## 13. Predicciones clippadas a ≥ 0

Demand y cash son no-negativas por definición. Sin clipping, una predicción negativa rompe la interpretación operativa. Aplicado en `modeling.assemble_predictions` y `cash_forecasting.predict_cash`.
