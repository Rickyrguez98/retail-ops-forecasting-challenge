# Review Checklist — Final

Resultado del Review Agent (Phase 8). 🟢 listo, 🟡 limitado por tiempo/datos, 🔴 pendiente.

## Entregables requeridos por el challenge

| Item | Status | Evidencia |
|------|--------|-----------|
| Repositorio reproducible | 🟢 | `make all` corre limpio desde `make clean` (verificado en este review) |
| `PROCESS.md` | 🟢 | Fases, herramientas, decisiones por fase |
| README con cómo correr | 🟢 | [`README.md`](../README.md) |
| Documentación de uso de IA | 🟢 | [`AI_USAGE.md`](../AI_USAGE.md) |
| Solución basada en datos útil al negocio | 🟢 | Demand + cash con buffer P90, ambas decisiones operativas reales |
| Validación de la solución | 🟢 | Métricas en test holdout (no usado para tuning); 19 tests deterministas |

## Calidad de código y arquitectura

| Item | Status | Nota |
|------|--------|------|
| Modularidad (src/ vs scripts/ vs notebooks/) | 🟢 | Lógica en `src/`, orquestación en `scripts/`, narrativa en `notebooks/` |
| Configuración externa (YAML, no hardcoded paths) | 🟢 | `configs/config.yaml` + `configs/model_config.yaml` |
| Logging configurable | 🟢 | `configs/logging.yaml` + `utils.setup_logging` |
| Versiones pinneadas | 🟢 | `requirements.txt` |
| Pre-commit hooks (black/isort/ruff) | 🟢 | `.pre-commit-config.yaml` (no se requiere instalar para pasar tests) |
| `pyproject.toml` con tooling moderno | 🟢 | Black, isort, ruff, pytest configurados |
| Reproducibilidad de semillas | 🟢 | `utils.set_seed` + `configs/config.yaml::seed: 42` |
| Estructura de paquete instalable | 🟢 | `src/retail_ops_forecasting/` |

## Rigor estadístico

| Item | Status | Nota |
|------|--------|------|
| Split temporal (no random) | 🟢 | Verificado por test |
| Validación incluye eventos para tuning | 🟢 | Val 2023-10 a 2023-11 incluye Buen Fin |
| Test holdout intacto | 🟢 | Test 2023-12 a 2024-02; no usado para tuning |
| Walk-forward CV disponible | 🟢 | `splits.walk_forward_folds` (no ejercitado en pipeline default; documentado en DECISIONS) |
| Baselines como ciudadanos de primera | 🟢 | 3 baselines registrados con MLflow antes del ML model |
| Métrica robusta a ceros | 🟢 | WAPE primaria; sMAPE secundaria |
| Bias reportado | 🟢 | Asimetría val/test documentada honestamente |

## Anti-leakage

| Item | Status | Nota |
|------|--------|------|
| `replenishment_signal` excluida | 🟢 | Documentada en DECISIONS §5 |
| Same-day components de targets excluidos | 🟢 | `amount_cash`, `amount_card`, `cash_transactions`, etc. |
| Lags computados con `shift(>=1)` agrupado por entidad | 🟢 | Test crítico: `test_rolling_mean_uses_only_past` |
| Rolling stats shifted before rolling | 🟢 | Test crítico: `test_lag_features_do_not_use_same_day_target` |
| Lista de leakage en config (auditable) | 🟢 | `configs/config.yaml::leakage_blocklist` |
| Test que verifica que blocklist se respeta | 🟢 | `test_select_feature_columns_drops_blocklist` |

## Experiment tracking

| Item | Status | Nota |
|------|--------|------|
| MLflow integrado | 🟢 | `src/tracking.py` con fallback JSON sidelog |
| Cada run tiene params + métricas + tags | 🟢 | Verificable en `./mlruns` o sidelog |
| Experimentos separados (demand vs cash) | 🟢 | `demand_forecasting` y `cash_forecasting` |
| Resumen automático generado | 🟢 | `reports/experiment_summary.md` vía `make report` |

## Documentación

| Item | Status | Nota |
|------|--------|------|
| README claro y conciso | 🟢 | TL;DR + cómo correr + dónde leer más |
| PROCESS.md con fases y herramientas | 🟢 | Incluye qué se quedó fuera por tiempo |
| AI_USAGE.md con transparencia | 🟢 | Explícito sobre qué generó IA y qué validé yo |
| DECISIONS.md con trade-offs | 🟢 | 12 decisiones documentadas con alternativas |
| Model card | 🟢 | Estilo Google con métricas, limitaciones, equidad |
| Final report con recomendaciones de negocio | 🟢 | 7 secciones incluyendo limitaciones y siguiente iteración |
| AGENT_WORKFLOW.md (orquestación) | 🟢 | `.claude/AGENT_WORKFLOW.md` |
| Diccionario de datos preservado | 🟢 | `data/raw/data_dictionary.md` |

## Tests

| Item | Status | Nota |
|------|--------|------|
| Tests pasan limpios | 🟢 | 19 passed, 5 warnings (deprecation upstream, no de mi código) |
| Anti-leakage tests críticos | 🟢 | `test_features.py` cubre lag y rolling |
| Splits tests | 🟢 | `test_splits.py` cubre disjuntez y orden temporal |
| Metric tests con valores hand-checked | 🟢 | `test_metrics.py` |
| Tests de cash buffer | 🟢 | `test_cash_forecasting.py` |
| Data validation tests | 🟢 | `test_data_validation.py` |
| Tests deterministas (no requieren CSVs reales) | 🟢 | Fixtures sintéticos en `conftest.py` |

## Git hygiene

| Item | Status | Nota |
|------|--------|------|
| Conventional commits | 🟢 | `chore:`, `feat:`, `exp:`, `analysis:`, `docs:` |
| Commits atómicos | 🟢 | 7 commits, cada uno con scope claro |
| Mensajes de exp con métricas | 🟢 | E.g. `exp(demand): HGB v1 — WAPE val 26.2%` |
| No commits de secrets / artefactos grandes | 🟢 | `.gitignore` cubre `data/interim`, `data/processed`, `models`, `mlruns` |

## Limitaciones honestas (🟡)

| Item | Status | Razón |
|------|--------|-------|
| Tuning sistemático de hiperparámetros | 🟡 | Default + early stopping; ~1-2 pp WAPE adicional posible |
| LightGBM benchmark | 🟡 | HGB suficiente; dejado como futura iteración |
| Recursive forecasting T+7 / T+28 | 🟡 | Pipeline modela T+1; recursivo no evaluado |
| Optimización de denominaciones de cash | 🟡 | No hay datos de denominaciones ni balance de apertura |
| Conformal prediction para cash | 🟡 | Buffer P90 empírico fue elegido por simplicidad |
| Quantile regression como segundo head para cash | 🟡 | Documentado en `final_report.md §7` |
| Análisis por tienda individual | 🟡 | Coverage por tienda generado pero no se modelan separadamente |
| Stripping de outputs de notebooks | 🟡 | Outputs comprometidos para conveniencia del reviewer; en CI se ejecutarían frescos |

## Conclusión

Estado: **listo para entrega**. Reproduce desde clean en ~30 s end-to-end. Tests pasan.
Documentación completa con números reales (no placeholders). Decisiones técnicas documentadas
con alternativas. Limitaciones identificadas y honestas.

Si el reviewer quiere reproducir:
```bash
git clone <repo>
cd retail-ops-forecasting-challenge
make install
make test          # 19 tests, ~1s
make all           # end-to-end pipeline, ~30s
```
