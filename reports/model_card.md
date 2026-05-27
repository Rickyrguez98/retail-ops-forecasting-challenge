# Model Card — Retail Operations Forecasting

Formato inspirado en [Google Model Cards](https://modelcards.withgoogle.com/about).

## Detalles del modelo

| | |
|--|--|
| **Versión** | 0.1.0 |
| **Modelo oficial elegido** | LightGBM tuneado con Optuna (TPE Bayesian, 30 trials) |
| **Modelo fallback** | Histogram-based Gradient Boosting Regressor (`sklearn.ensemble.HistGradientBoostingRegressor`) |
| **Capa post-hoc opcional** | *Event stress adjustment* (regla de negocio, ver §Métricas) |
| **Entrenado por** | Candidate, Walmart Senior DS Challenge |
| **Fecha** | 2026-05-26 |
| **Licencia** | Proprietary (uso interno del challenge) |
| **Punto de contacto** | Repositorio del candidato |

Dos modelos hermanos comparten arquitectura y pipeline de features:

| Modelo | Granularidad | Target | Features | Filas train |
|--------|--------------|--------|----------|-------------|
| Demand | store × category × día | `total_transactions` | 40 (lags, rolling, calendar, store metadata) | 117,576 (post drop NA) |
| Cash   | store × día | `amount_cash` | 41 (cash lags, rolling, calendar, store metadata) | 19,596 (post drop NA) |

## Uso previsto

**Casos de uso:**
- Demand: planeación semanal de staffing y capacidad de checkout.
- Cash: recomendación operativa de cuánto efectivo cargar por sucursal/día.

**Usuarios objetivo:** equipos de operaciones de tienda, planeación de staffing, tesorería operativa.

**Casos de uso fuera de alcance:**
- Forecasting recursivo a > 7 días sin re-training.
- Forecasting para tiendas/categorías nuevas sin historial.
- Optimización de denominaciones de efectivo (no hay datos de denominaciones).
- Decisiones financieras o regulatorias.

## Métricas de desempeño (test holdout, 2023-12-01 → 2024-02-29)

> **Política de reporte**: las métricas oficiales out-of-sample del modelo son las
> **clean holdout** (sección A). La sección B reporta un **escenario operativo post-hoc**
> con una capa de regla de negocio (*event stress adjustment*). Mezclar los dos números
> en una sola tabla escondería de qué viene cada mejora.

### A — Clean holdout (modelo oficial, sin ajuste post-hoc)

#### Demand

| Métrica | Mejor baseline | HGB base | **LightGBM tuneado** | Δ vs baseline |
|---------|---------------:|---------:|---------------------:|--------------:|
| WAPE    | 0.323 (lag-7)  | 0.296    | **0.293**            | **−3.0 pp** (−9.4 % rel.) |
| MAE     | 228.7          | 209.4    | **207.7**            | −21.0 tx/día |
| RMSE    | 548.0          | 467.9    | **462.4**            | −85.6 |
| sMAPE   | 0.287          | 0.277    | **0.274**            | −0.013 |
| Bias    | −16.2          | −26.9    | **−27.5**            | leve sub-forecast |

WAPE val (incluye Buen Fin): HGB base **0.262**, LightGBM tuneado **0.258**.

#### Cash

| Métrica | Valor (test) |
|---------|-------------:|
| WAPE    | **0.153** |
| MAE     | MXN 71,087 |
| RMSE    | MXN 200,007 |
| sMAPE   | 0.137 |
| Bias    | +14,626 (under-forecast → buffer compensa) |
| **Coverage P90 (recomendado ≥ actual)** | **0.946** |

### B — Escenario operativo post-hoc (con *event stress adjustment*)

Capa de regla de negocio aplicada al modelo base: factor de Buen Fin aprendido del set
de validación; factores Dic 24-25 y Dic 31 son defaults basados en estacionalidad
publicada del retail mexicano. **NO calibrados contra test.** Antes de producción
deben re-calibrarse con múltiples años previos de eventos comparables.

| Modelo + capa post-hoc | WAPE val | WAPE test (escenario) | MAE test |
|--------|---------:|---------------------:|---------:|
| HGB + event stress adjustment | 0.179 | 0.249 | 176.1 |
| LightGBM + event stress adjustment | 0.178 | **0.247** | **174.8** |

> Estos números **no son** el desempeño out-of-sample oficial del modelo. Son el
> resultado del modelo + una capa de regla de negocio aplicada a fechas conocidas
> de estrés navideño.

### Desempeño por segmento (escenario operativo §B)

Los breakdowns por segmento se reportan sobre el escenario operativo (modelo base +
capa post-hoc) porque representan el comportamiento del sistema en producción si
se decidiera activar la capa.

**Demand WAPE por evento:** payday **0.20**, weekend **0.23**, regular **0.25**, navidad **0.29**, holiday **0.39** (n=1,920, alta varianza).
La capa post-hoc redujo holiday de 0.65 → 0.39 (−26 pp) y weekend de 0.31 → 0.23 (−8 pp).

**Demand WAPE por categoría:** Abarrotes **0.24**, Bebidas **0.24**, Cuidado_Personal **0.26**, Hogar **0.26**, Ropa **0.27**, Electronica **0.29** (categoría más pequeña, mayor relativo).

**Demand WAPE por formato:** Bodega **0.24**, Supercenter **0.25**, Express **0.26** — prácticamente parejo.

**Cash WAPE por evento:** regular **0.11**, navidad **0.12**, payday **0.13**, weekend **0.19**, holiday **0.21**.

**Cash WAPE por formato:** Express **0.13**, Bodega **0.14**, Supercenter **0.18**.

## Datos de entrenamiento

- **Fuente:** dataset proporcionado por el challenge — 80 tiendas × 6 categorías × 425 días.
- **Split:**
  - Train: 2023-01-01 → 2023-09-30 (~131k filas demand, ~22k filas cash)
  - Val: 2023-10-01 → 2023-11-30 (incluye Buen Fin para tuning)
  - Test: 2023-12-01 → 2024-02-29 (held-out; incluye Navidad)
- **Filas excluidas:** cualquiera con `target` NaN. Para cash: ~5.9% de filas store-day excluidas por nulos en `amount_cash`.
- **Filas dropeadas por warm-up de lags:** primeras 28 filas de cada serie (necesario para `lag_28` y `rmean_28`).

## Features

40 features para demand, 41 para cash. Categorías de features:

1. **Calendar interactions** (`dow_sin/cos`, `month_sin/cos`, `is_payday × is_weekend`, `store_age_years`).
2. **Lags del target**: 1, 7, 14, 28.
3. **Rolling stats**: mean 7 y 28, std 28 (todos con `shift(>=1)` para anti-leakage).
4. **Calendar flags**: `is_holiday`, `is_payday`, `is_weekend`, `is_navidad_season`, `is_buen_fin`, `is_semana_santa`, `is_event`.
5. **Store metadata**: `store_format`, `region`, `size_sqm`, `num_checkouts`, `opening_year`, `socioeconomic_level`, `has_pharmacy`, `has_fuel_station`.
6. **Cash-specific** (sólo modelo de cash): `cash_share_lag_1`, `total_transactions` lagged como feature auxiliar.

Categóricas codificadas a int via `encode_categoricals` (sklearn 1.2.2 no soporta strings nativos).

## Leakage controls

| Columna | Excluida | Razón |
|---------|----------|-------|
| `replenishment_signal` | ✅ | Diccionario indica que se calcula de la demanda observada |
| `cash_transactions` | ✅ | Componente same-day del target de cash |
| `amount_cash` (demand) / como feature | ✅ | Mismo evento contemporáneo |
| `amount_card`, `card_transactions`, `amount_total` | ✅ | Reconstruirían el target de cash |
| `units_sold`, `avg_ticket` | ✅ | Output contemporáneo de la demanda |

El test `test_select_feature_columns_drops_blocklist` verifica que ninguna entra al modelo.

## Componentes implementados

1. **Anti-leakage controls**: lag y rolling features con `shift(>=1)`, leakage blocklist en config, 3 tests críticos verifican el invariante.
2. **Bayesian hyperparameter optimization** (Optuna TPE): 30 trials sobre 9 hiperparámetros de LightGBM. El modelo tuneado es la elección oficial; HGB queda como fallback con desempeño equivalente.
3. ***Event stress adjustment* post-hoc** (opcional, off por default sería conservador): capa de regla de negocio sobre días pico navideños. Documentada como escenario operativo, **no como performance out-of-sample del modelo**. Antes de producción debe re-calibrarse con 2-3 años previos.
4. **Forecast horizon T+1** con pipeline reproducible: `make all` regenera todos los artefactos en ~30s end-to-end.
5. **Cash buffer P90 por tienda**: regla operativa transparente, supera el target de coverage 90 % (resultado: 94.6 %).

## Consideraciones éticas y de equidad

- **Tipo de datos:** operativos (no personales). No hay PII en el dataset.
- **Sesgos potenciales:** las recomendaciones de cash y staffing podrían favorecer estructuralmente a Supercenters (mayor volumen, mejor señal). Comparar coverage por formato confirma que **Supercenters tienen WAPE de cash más alto (0.18) que Express (0.13)** — el modelo es ligeramente peor donde más importa. Mitigación: tuning específico por formato.
- **Uso responsable:** la recomendación de cash es un **proxy** de planeación. Decisiones operativas finales requieren input humano (recolección, denominaciones).

## Mantenimiento

- **Retraining cadence sugerido:** mensual, después del cierre del periodo Buen Fin / Navidad para incorporar los eventos.
- **Drift monitoring:** trackear WAPE rolling 28-day por región y formato; alarmar si excede 0.40 (demand) o 0.25 (cash).
- **Tests automáticos:** `tests/test_features.py::test_lag_features_do_not_use_same_day_target` debe pasar en cada deployment.
