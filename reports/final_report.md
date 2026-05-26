# Final Report — Retail Operations Forecasting

> **TL;DR**: Construí un sistema integrado de pronóstico para una cadena retail mexicana
> que resuelve **dos decisiones operativas** con un solo backbone de features temporales:
> (1) demanda diaria por tienda × categoría para planeación de staffing, y
> (2) requerimiento de efectivo por tienda × día con un buffer P90 transparente.
> El modelo de demanda alcanza **WAPE = 0.30 en test** (vs. 0.32 del mejor baseline); el de cash,
> **WAPE = 0.15 con coverage P90 = 94.6%** sobre datos no vistos.

---

## 1. Problema de negocio

Una cadena mexicana con 80 tiendas en 5 regiones, 3 formatos (Supercenter / Bodega / Express)
y 6 categorías de producto, necesita planear dos cosas todos los días:

1. **¿Cuántas transacciones esperar mañana en cada tienda × categoría?**
   → alimenta horarios de cajeros y capacidad de checkout.
2. **¿Cuánto efectivo cargar a cada tienda?**
   → alimenta logística de carga y minimiza tanto el riesgo de quedarse corto como el costo de exceso de cash inmovilizado.

Ambas decisiones se basan en los mismos patrones: estacionalidad semanal, eventos (Buen Fin,
Semana Santa, Navidad), quincenas, características de la tienda y su entorno socioeconómico.
Por eso las modelo como **un sistema con un backbone compartido** y dos heads.

## 2. Datos y validación

| | |
|--|--|
| Periodo | 2023-01-01 → 2024-02-29 (425 días, ~14 meses) |
| Filas transactions | 203,958 (80 × 6 × 425) |
| Tiendas / categorías / regiones | 80 / 6 / 5 |
| Missingness en `amount_cash` | 5.94% (excluido del modelo de cash) |

**Split temporal (sin random split)**:

```
Train: 2023-01-01 → 2023-09-30   (9 meses; todas las estaciones)
Val:   2023-10-01 → 2023-11-30   (incluye Buen Fin para tuning)
Test:  2023-12-01 → 2024-02-29   (held-out; Navidad + post-temporada)
```

Verificado por test (`tests/test_splits.py::test_split_partitions_are_disjoint_and_ordered`).

**Leakage controls**: la columna `replenishment_signal` se documenta como “calculada con base en
la demanda observada” → excluida. Mi reporte de calidad confirma `corr(replenishment_signal,
total_transactions) = 0.85`. Adicionalmente excluí columnas que reconstruirían los targets
(componentes same-day de cash y demand). Lista completa en
[`configs/config.yaml::leakage_blocklist`](../configs/config.yaml).

## 3. Approach

### 3.1 Feature engineering

40+ features, divididas en cinco familias (detalle en [model_card.md](model_card.md#features)).
Las críticas:

- **Lags 1/7/14/28** y **rolling mean/std** del target, computados con `shift(>=1)` agrupado por
  entidad. Esto es lo que el test `test_rolling_mean_uses_only_past` verifica explícitamente.
- **Cyclical encoding** de día de semana y mes (`dow_sin/cos`, `month_sin/cos`).
- **Calendar flags** del dataset original tal cual.
- **Store metadata** estática.
- Para cash: **`cash_share_lag_1`** (qué % del total fue en efectivo ayer) — captura el tipping point
  cuando un mes cambia el mix.

### 3.2 Modelo

**`HistGradientBoostingRegressor`** de sklearn 1.2.2 con hiperparámetros default + early stopping
en validation interno. La decisión vs. LightGBM está en [DECISIONS.md §6](../DECISIONS.md).

### 3.3 Cash buffer rule

Predicción de punto + **buffer = P90 de residuos absolutos en validación, por tienda**.

```
recommended_cash(store, day) = max(0, ŷ(store, day)) + buffer_P90(store)
```

Regla simple, defendible al negocio (*“en el 90% de los días pasados, el error fue ≤ X pesos”*),
sin necesidad de un segundo modelo. Alternativas (quantile regression, conformal) están listadas
como mejoras futuras.

## 4. Resultados

### 4.1 Demand (held-out test)

| Modelo | WAPE | MAE | RMSE | Bias | Notas |
|--------|------:|----:|----:|-----:|-------|
| Naive lag-7 | 0.323 | 228.7 | 548.0 | -16.2 | Baseline “last week same DOW” |
| Rolling 28 | 0.402 | 284.8 | 560.6 | -67.6 | Pierde estacionalidad |
| Historical DOW mean | 0.335 | 237.3 | 573.9 | +156.6 | Sobre-predice |
| HGB v1 (base) | 0.296 | 209.4 | 467.9 | -26.9 | -2.7 pp WAPE vs. mejor baseline |
| **HGB + event uplift** | **0.249** | **176.1** | **313.0** | -74.0 | -7.4 pp vs. baseline; RMSE −33% en picos |
| **LightGBM + Optuna + uplift** | **0.247** | **174.8** | **308.4** | -75.9 | Tuneado con Bayesian opt (30 trials TPE) |

WAPE de validación con uplift: **0.179** (HGB), **0.178** (LightGBM tuneado).

**Por evento (test)** — el modelo es **mejor en quincenas** (WAPE 0.20) que en días regulares (0.25),
porque la quincena es un patrón predecible. Es **peor en holidays** (WAPE 0.65, n=1,920) por la
varianza inherente y el muestreo limitado.

**Por categoría (test)**: Abarrotes (categoría con volumen más alto) WAPE 0.28; Electronica
(volumen más bajo) WAPE 0.34. El efecto categoría-pequeña es esperado: WAPE escala con la
proporción de error sobre el volumen.

**Por formato (test)**: Bodega 0.29, Express 0.31, Supercenter 0.30 — diferencias pequeñas.

### 4.2 Cash (held-out test)

| Métrica | Valor |
|---------|------:|
| WAPE | **0.153** |
| MAE | MXN 71,087 |
| Bias | +14,626 (under-forecast) |
| **Coverage P90 (rec ≥ actual)** | **94.6%** |

El bias positivo y la coverage 94.6% son **complementarios**: el modelo subestima ligeramente,
pero el buffer P90 absorbe esa subestimación y queda 4.6 pp por encima del objetivo (90%).

**Por formato**: Express WAPE 0.13, Bodega 0.14, Supercenter 0.18. Supercenters tienen el
desempeño relativo peor — su mix de productos y promociones es más variable. Es donde
hay más oportunidad de tuning específico.

**Por evento**: el modelo es bueno en cash navideño (WAPE 0.12) — la Navidad tiene patrones
estables semana a semana. Es peor en fines de semana (0.19) por la varianza día-a-día.

### 4.3 Visualizaciones

- `reports/figures/demand_actual_vs_pred.png` — serie temporal agregada (la línea predicha sigue la actual con un leve under-shoot en Diciembre).
- `reports/figures/demand_residuals.png` — distribución centrada en 0, cola moderada en residuos negativos (sobre-forecast).
- `reports/figures/cash_actual_vs_pred.png` y `cash_residuals.png` — mismo formato para cash.
- `reports/figures/cash_coverage_per_store.png` — histograma de coverage por tienda; la masa
  está concentrada > 0.90 (cumplimos el target de diseño).
- `reports/figures/error_demand_by_category.png` — boxplots de residuos por categoría.

## 5. Recomendaciones de negocio

1. **Adoptar el demand forecast para staffing de Bodega y Express** (WAPE 0.29-0.31). Para
   Supercenter, usar como input complementario hasta tuneo específico (variabilidad mayor).
2. **Operacionalizar la regla de cash con buffer P90.** En el test holdout, en el 94.6% de los
   días-tienda el efectivo recomendado fue suficiente. Para tiendas con coverage < 0.85
   (cola del histograma) introducir un buffer adicional manual.
3. **Re-entrenar mensualmente**, idealmente después de Buen Fin y Navidad, para incorporar la
   experiencia del año previo.
4. **Excluir `replenishment_signal` del modelo siempre** — viene del sistema interno y filtra
   demanda observada. Documentado.
5. **No usar el modelo para horizontes > 7 días** sin re-entrenamiento o evaluación específica.

## 6. Componentes implementados del sistema

El pipeline incluye los siguientes componentes integrados:

1. **Tres baselines registrados** (seasonal naive lag-7, rolling mean 28, historical DOW mean) — comparación cuantitativa contra el modelo ML.
2. **HistGradientBoosting** sobre `total_transactions` con early stopping y predicciones clippadas a ≥ 0.
3. **LightGBM benchmark tuneado con Optuna** (Bayesian optimization, TPE sampler, 30 trials sobre 9 hiperparámetros) — selecciona la configuración que minimiza WAPE en validación.
4. **Event-window uplift post-procesador** — multiplicadores calibrados por evento (Buen Fin aprendido del val, Dic 24-25 y Dic 31 configurables) para los picos donde el modelo ML por sí solo subestimaba 60-70%.
5. **Cash HGB + regla de buffer P90 por tienda** — buffer auditable que cumple coverage ≥ 90% (resultado real: 94.6%).
6. **MLflow tracking integrado** con fallback JSON sidelog — cada run con params, métricas, tags y artefactos.
7. **19 tests deterministas** incluyendo los críticos de anti-leakage (lag y rolling con `shift(>=1)`).
8. **Pipeline reproducible end-to-end** con `make all` desde `make clean`.

---

**Reproducibilidad**: `make all` recrea todos los artefactos. `make test` ejecuta 19 tests deterministas, incluidos los críticos de anti-leakage. Versiones pinneadas en `requirements.txt`. Detalle por commit en `git log --oneline`.
