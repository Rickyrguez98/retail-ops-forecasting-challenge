# Review Checklist — Senior-Level Final Review

Auto-evaluación de la entrega contra las 13 dimensiones que un revisor senior
de Data Science / ML Engineering típicamente audita. Cada item tiene status,
evidencia concreta (archivo o número) y, donde aplica, la limitación honesta.

Status: 🟢 implementado · 🟡 parcial / con caveat · 🔴 no aplica o pendiente

---

## 1. Business framing

| Item | Status | Evidencia |
|------|--------|-----------|
| Problema de negocio articulado | 🟢 | Demand para staffing y cash para carga operativa, ambas decisiones reales — descritas en [`SUBMISSION.md`](../SUBMISSION.md) y [`final_report.md §1`](final_report.md) |
| Conexión entre métrica y decisión | 🟢 | WAPE → ¿cuántas transacciones por error? (208 MAE) → ¿cuántos cajeros de más/menos? Coverage P90 → ¿en qué fracción de días la carga fue suficiente? |
| Recomendaciones de negocio accionables | 🟢 | 6 recomendaciones operativas en [`final_report.md §5`](final_report.md), incluyendo política de rollout del modelo y la capa post-hoc |

## 2. Methodological credibility

| Item | Status | Evidencia |
|------|--------|-----------|
| Distinción explícita modelo "oficial" vs ajustes operativos | 🟢 | Tabla A (clean holdout) vs Tabla B (post-hoc) en [`final_report.md §4.1`](final_report.md) y [`model_card.md`](model_card.md) |
| Selección de modelo con criterios cuantitativos | 🟢 | LightGBM tuneado vs HGB vs baselines en [`final_report.md §4.3`](final_report.md): gana en WAPE, MAE y RMSE consistentemente |
| Significado del WAPE explicado en términos de negocio | 🟢 | "29.3 % del volumen real" + comparativa con rango típico retail (0.20-0.40) — [`final_report.md §4.3`](final_report.md) |
| Hyperparameter tuning Bayesiano | 🟢 | Optuna TPE, 30 trials, 9 hiperparámetros, optimiza WAPE en **val** (no test) — [`DECISIONS.md §6`](../DECISIONS.md) |

## 3. Leakage prevention

| Item | Status | Evidencia |
|------|--------|-----------|
| `replenishment_signal` identificada y excluida | 🟢 | Diccionario indica que se calcula de demanda observada; documentado en [`DECISIONS.md §5`](../DECISIONS.md) |
| Blocklist de leakage en config | 🟢 | [`configs/config.yaml::leakage_blocklist`](../configs/config.yaml) cubre 8 columnas |
| Lag features con `shift(>=1)` agrupado | 🟢 | `src/retail_ops_forecasting/features.py` |
| Rolling stats shifted antes del rolling | 🟢 | Mismo módulo, función `_group_rolling_mean` |
| Tests automáticos del invariante | 🟢 | `test_lag_features_do_not_use_same_day_target`, `test_rolling_mean_uses_only_past`, `test_select_feature_columns_drops_blocklist` |
| Capa post-hoc NO calibrada contra test | 🟢 | Buen Fin aprendido de val; Dec 24-25 / Dec 31 son business-rule defaults — explícito en [`configs/config.yaml`](../configs/config.yaml) y [`DECISIONS.md §7`](../DECISIONS.md) |

## 4. Validation strategy

| Item | Status | Evidencia |
|------|--------|-----------|
| Split temporal (no random) | 🟢 | Train 2023-01 → 09 / Val 2023-10 → 11 / Test 2023-12 → 2024-02 |
| Val incluye eventos para tuning legítimo | 🟢 | Val cubre Buen Fin (Nov 2023) |
| Test holdout incluye Navidad | 🟢 | Test cubre Dic 2023 + Ene–Feb 2024 |
| Test holdout evaluado **una sola vez** | 🟢 | Tuning de modelo y de capa Buen Fin ocurre sobre val; Dec 24-25 / Dec 31 son defaults pre-establecidos |
| Walk-forward CV disponible | 🟡 | `splits.walk_forward_folds` implementada y testada; no se ejercitó en pipeline default (sería mejora para producción) |
| Test que valida la disjuntez de splits | 🟢 | `test_split_partitions_are_disjoint_and_ordered` |

## 5. Clean holdout vs post-hoc adjustment separation

| Item | Status | Evidencia |
|------|--------|-----------|
| Métricas oficiales = clean holdout sin ajustes | 🟢 | Headline en [`README.md`](../README.md) y [`final_report.md TL;DR`](final_report.md): LightGBM tuneado WAPE test = **0.293** (sin uplift) |
| Capa post-hoc reportada como escenario, en tabla aparte | 🟢 | [`final_report.md §4.1.B`](final_report.md), [`model_card.md §B`](model_card.md) |
| Disclaimer explícito sobre que el escenario ≠ performance out-of-sample | 🟢 | Tres reports lo dicen literalmente |
| Política de rollout documentada (start con uplift off) | 🟢 | [`final_report.md §5.3`](final_report.md), [`DECISIONS.md §7`](../DECISIONS.md) |
| Calibración futura de la capa especificada | 🟢 | "2-3 años previos de eventos comparables, no contra test" |

## 6. Cash forecasting operational value

| Item | Status | Evidencia |
|------|--------|-----------|
| Granularidad y target documentados | 🟢 | store × day, target `amount_cash` — [`DECISIONS.md §3`](../DECISIONS.md) |
| Regla de buffer simple y auditable | 🟢 | `recommended = max(0, ŷ) + P90_residuos_val(store)` |
| Coverage objetivo definido a priori | 🟢 | Target ≥ 90 %; resultado test = 94.6 % |
| Bias y coverage reportados como complementarios | 🟢 | Bias +14,626 (sub-forecast) absorbido por buffer; explicado en [`final_report.md §4.2`](final_report.md) |
| Coverage por tienda (no solo agregado) | 🟢 | Figura `cash_coverage_per_store.png` + breakdowns por formato/región/evento |

## 7. Experiment tracking

| Item | Status | Evidencia |
|------|--------|-----------|
| MLflow integrado con fallback | 🟢 | `src/retail_ops_forecasting/tracking.py` con JSON sidelog si MLflow falla |
| Experimentos separados | 🟢 | `demand_forecasting`, `cash_forecasting`, `evaluation` |
| Params + métricas + tags por run | 🟢 | Verificable en `./mlruns` o `reports/*_tracking_sidelog.json` |
| Model Registry poblado | 🟢 | `demand_hgb v1`, `demand_lgb v1`, `cash_hgb v1` con signature de entrada/salida |
| Dataset lineage | 🟢 | 9 datasets logueados (train/val/test × demand + cash) con source y schema |
| Artifacts por run | 🟢 | Model files, predicciones CSV, figuras y breakdowns adjuntos al run |
| UI accesible vía Docker | 🟢 | `docker compose up mlflow` → http://localhost:5001 |

## 8. Reproducibility

| Item | Status | Evidencia |
|------|--------|-----------|
| Seeds fijos en config | 🟢 | `seed: 42` en [`configs/config.yaml`](../configs/config.yaml); `utils.set_seed` lo aplica |
| Versiones pinneadas | 🟢 | `requirements.txt` con `==` para libs críticas y `>=` para tooling |
| Pipeline end-to-end con un comando | 🟢 | `make all` o `docker compose run --rm pipeline` |
| Tests deterministas (no requieren CSVs raw) | 🟢 | Fixtures sintéticos en `conftest.py` |
| Verificación local ↔ Docker idéntica | 🟢 | Métricas byte-a-byte iguales (comprobado y commiteado anteriormente) |
| Repo local ↔ GitHub byte-a-byte sincronizado | 🟢 | Tree hash SHA-256 idéntico verificado |

## 9. CI

| Item | Status | Evidencia |
|------|--------|-----------|
| GitHub Actions workflow | 🟢 | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) corre en push y PR |
| Lint en CI (black + isort + ruff) | 🟢 | Tres pasos separados; CI falla si alguno falla |
| Tests en CI (pytest) | 🟢 | Step final `pytest -ra` |
| Cache de pip para builds rápidos | 🟢 | `cache: pip` con key sobre `requirements.txt` |
| Badge de status en README | 🟢 | Renderiza el estado actual del workflow |
| `make lint` localmente = lo que corre CI | 🟢 | Mismos tres comandos exactos |

## 10. Code modularity

| Item | Status | Evidencia |
|------|--------|-----------|
| Separación `src/` vs `scripts/` vs `notebooks/` | 🟢 | Lógica en `src/retail_ops_forecasting/`, orquestación en `scripts/`, narrativa en `notebooks/` |
| Paquete instalable | 🟢 | `pyproject.toml` + `pip install -e .` |
| Configuración externa (YAML, no hardcode) | 🟢 | `configs/config.yaml`, `configs/model_config.yaml`, `configs/logging.yaml` |
| Funciones puras vs side-effects separados | 🟢 | `features.py`, `metrics.py`, `splits.py` puros; tracking e I/O aislados |
| Logging configurable | 🟢 | `configs/logging.yaml` + `utils.setup_logging` |
| Sin cyclic imports / sin scripts importables como módulos | 🟢 | `scripts/` están en `if __name__ == "__main__"` |

## 11. Tests

| Item | Status | Evidencia |
|------|--------|-----------|
| Tests pasan limpios | 🟢 | 19/19 passed (`make test` o CI) |
| Anti-leakage tests críticos | 🟢 | `test_features.py` cubre lag y rolling |
| Splits tests | 🟢 | `test_splits.py` cubre disjuntez y orden temporal |
| Metric tests con valores hand-checked | 🟢 | `test_metrics.py` |
| Cash buffer tests | 🟢 | `test_cash_forecasting.py` (incluye coverage) |
| Data validation tests | 🟢 | `test_data_validation.py` |
| Determinismo (mismo input → mismo output) | 🟢 | Sin tests flaky observados; seeds fijos |

## 12. Documentation

| Item | Status | Evidencia |
|------|--------|-----------|
| README como entrada al proyecto | 🟢 | TL;DR, instalación, comandos, estructura — [`README.md`](../README.md) |
| Submission summary (mapa al revisor) | 🟢 | [`SUBMISSION.md`](../SUBMISSION.md) |
| Process log con fases y herramientas | 🟢 | [`PROCESS.md`](../PROCESS.md) |
| Decisiones técnicas con alternativas | 🟢 | 13 decisiones en [`DECISIONS.md`](../DECISIONS.md) |
| Model card (Google style) | 🟢 | [`model_card.md`](model_card.md) con métricas, limitaciones, equidad, mantenimiento |
| Final report con recomendaciones | 🟢 | [`final_report.md`](final_report.md) — incluye §4.3 "Por qué elegimos LightGBM tuneado y qué significa un WAPE de 0.293" |
| Documentación consistente en español | 🟢 | Todos los `.md` en español; términos técnicos en inglés cuando son nombres propios (WAPE, holdout, staffing) |
| Diccionario de datos preservado | 🟢 | `data/raw/data_dictionary.md` |

## 13. AI usage transparency

| Item | Status | Evidencia |
|------|--------|-----------|
| Archivo dedicado | 🟢 | [`AI_USAGE.md`](../AI_USAGE.md) |
| Lista de qué generó IA y qué validé yo | 🟢 | Tabla por componente |
| Decisiones críticas tomadas por mí (no por IA) | 🟢 | Detección de `replenishment_signal` como leakage, estrategia de splits, regla de buffer P90, separación clean/post-hoc |
| Outputs del LLM rechazados / corregidos | 🟢 | Documentados explícitamente |
| Sin pretender que el código es 100 % mío | 🟢 | Honestidad explícita sobre el uso de Claude como par programador |

---

## Resumen ejecutivo

| Dimensión | Status |
|-----------|--------|
| 1. Business framing | 🟢 |
| 2. Methodological credibility | 🟢 |
| 3. Leakage prevention | 🟢 |
| 4. Validation strategy | 🟢 (walk-forward CV implementada pero no ejercitada en pipeline default) |
| 5. Clean holdout vs post-hoc separation | 🟢 |
| 6. Cash forecasting operational value | 🟢 |
| 7. Experiment tracking | 🟢 |
| 8. Reproducibility | 🟢 |
| 9. CI | 🟢 |
| 10. Code modularity | 🟢 |
| 11. Tests | 🟢 |
| 12. Documentation | 🟢 |
| 13. AI usage transparency | 🟢 |

**Conclusión**: el repositorio está listo para entrega. Métricas oficiales sobre test
holdout limpio (sin ajustes post-hoc):

- Demanda: WAPE = **0.293** (LightGBM tuneado, modelo elegido) — −9.4 % rel. vs. baseline 0.323.
- Cash: WAPE = **0.153**, coverage P90 = **94.6 %** (target ≥ 90 %).
- 19/19 tests pasan localmente y en CI.
- 60+ commits incrementales con conventional-style messages.

### Cómo reproducir

```bash
git clone https://github.com/Rickyrguez98/retail-ops-forecasting-challenge.git
cd retail-ops-forecasting-challenge

# Opción A — Local
make install && make test && make lint && make all

# Opción B — Docker
docker compose build
docker compose run --rm tests       # 19/19 passed
docker compose run --rm pipeline    # equivale a `make all`
docker compose up mlflow            # UI en http://localhost:5001
```
