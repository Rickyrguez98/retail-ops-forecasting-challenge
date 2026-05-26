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

| Modelo | WAPE | MAE | Bias | Notas |
|--------|------:|----:|-----:|-------|
| Naive lag-7 | 0.323 | 228.7 | -16.2 | Baseline “last week same DOW” |
| Rolling 28 | 0.402 | 284.8 | -67.6 | Pierde estacionalidad |
| Historical DOW mean | 0.335 | 237.3 | +156.6 | Sobre-predice |
| **HGB v1** | **0.296** | **209.4** | -26.9 | -2.7 pp WAPE vs. mejor baseline |

WAPE de validación (incluye Buen Fin, condiciones más duras): **0.262**.

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

## 6. Limitaciones honestas

- **Sin tuning sistemático**. Defaults razonables; estimación: 1-2 pp WAPE adicionales son posibles con tuning + features adicionales (e.g., interacciones región × evento).
- **1 sola ocurrencia de Buen Fin / Navidad en train**. Limita generalización año-tras-año.
- **5.9% de filas store-day excluidas** por nulos en `amount_cash`. Métricas de cash representan el 94% del universo, no el 100%.
- **No hay optimización de denominaciones**: el modelo recomienda monto agregado, no qué billetes/monedas cargar. Requiere datos adicionales (balance de apertura, calendario de recolección).
- **Bias asimétrico val→test en demand** (val +112, test -27): el modelo subestima Buen Fin (sólo lo ve 1 vez en train). Documentado, aceptable para uso operativo con re-training.
- **Tests anti-leakage cubren las rutas críticas**, pero no son exhaustivos. Auditoría humana periódica recomendada.

## 7. Siguiente iteración

Si tuviera 2 semanas más:

1. **Quantile regression para cash** (modelo conjunto que predice P50 y P90), reemplazaría la regla de buffer empírico.
2. **LightGBM como benchmark** con tuning más extenso (Bayesian optimization).
3. **Recursive forecasting a T+7** y T+28, con evaluación de error acumulado.
4. **Modelos por formato** (un HGB por Supercenter, otro por Bodega, otro por Express) — la heterogeneidad de Supercenter justifica un modelo dedicado.
5. **Feature: interacciones categoría × evento** (uplifts diferenciados de Buen Fin por categoría).
6. **Análisis de tiendas con periodos vacíos**: caracterizar el patrón (¿es POS?, ¿es conectividad?), proponer imputación o flag.
7. **Pronóstico de `units_sold`** como modelo secundario (importante para reposición de inventario; complementa el de demanda).

---

**Reproducibilidad**: `make all` recrea todos los artefactos. `make test` ejecuta 19 tests deterministas, incluidos los críticos de anti-leakage. Versiones pinneadas en `requirements.txt`. Detalle por commit en `git log --oneline`.
