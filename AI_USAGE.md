# AI_USAGE — Transparencia de uso de IA

## TL;DR

Usé **Claude (Anthropic)** como asistente durante todo el desarrollo. Concretamente, dentro de
**Claude Code** (CLI) en modo agente. Todo el contenido — código, decisiones técnicas,
elección de modelo, framing del problema de negocio, interpretación de métricas — fue
revisado y aprobado por mí antes de quedar en el repo. No hay código autogenerado que no
haya inspeccionado y validado.

## Cómo lo usé

| Uso | Qué hizo Claude | Qué hice yo |
|-----|-----------------|-------------|
| **Scaffolding** | Generó el esqueleto de carpetas, `pyproject.toml`, `Makefile`, configuraciones YAML. | Validé que cada archivo cumple con los estándares que pedí (PEP 621, splits temporales correctos, leakage blocklist explícita). |
| **Módulos `src/`** | Escribió las primeras versiones de `data.py`, `validation.py`, `features.py`, `splits.py`, `modeling.py`, `cash_forecasting.py`, `metrics.py`, `tracking.py`, `reporting.py`. | Revisé cada función línea por línea. Las decisiones críticas (cómo se computa `rmean_7` con `shift(>=1)`, la regla de buffer P90, el contrato de leakage) las definí yo y Claude las implementó. |
| **Tests** | Generó tests anti-leakage, métricas y splits. | Verifiqué que los tests realmente fallan si se introduce leakage (probado removiendo el `shift(1)` en `_group_rolling_mean` — el test `test_rolling_mean_uses_only_past` rompe). |
| **Notebooks** | Estructura inicial de los 4 notebooks de EDA y análisis. | El análisis estadístico y las hipótesis a probar las definí yo. Claude ejecutó el código y resumió. |
| **Documentación** | Drafts de README, PROCESS, DECISIONS, model_card, final_report. | Edité, recorté, ajusté tono. La estructura de los documentos y qué información incluir fue decisión mía. |
| **Revisión cruzada** | Le pedí que actuara como reviewer senior y buscara bugs/anti-patterns. | Apliqué las sugerencias que tenían sentido; rechacé las que sobre-complicaban. |

## Lo que NO delegué

- **Framing del problema de negocio**: dos objetivos conectados (demanda + cash) — decisión mía después de inspeccionar los datos.
- **Detección de leakage**: identifiqué `replenishment_signal` como leakage leyendo el diccionario; Claude no lo señaló inicialmente.
- **Elección de WAPE como métrica primaria**: argumento de robustez ante ceros y alineación con planeación retail.
- **Estrategia de validación**: 9 meses train, 2 meses val con Buen Fin (para tuning), 3 meses test con Navidad.
- **Buffer rule**: P90 de residuos por tienda. Defendible y simple. Alternativas (quantile regression, conformal prediction) se descartaron por tiempo.
- **Interpretación de resultados**: las recomendaciones de negocio en `final_report.md` son mías.

## Prompts notables

- *“Build a feature pipeline where lag and rolling features cannot leak same-day target. Test it explicitly.”*
- *“Implement a cash forecasting pipeline with a transparent per-store P90 safety buffer. Add coverage as a metric.”*
- *“Act as a senior DS reviewer: what's wrong with this validation strategy? Be harsh.”*

## Modelo y versión

- **Claude Code** CLI con modelo Opus 4.7.
- Las conversaciones no quedaron en el repo. Si interesa, los prompts y la línea de razonamiento se pueden reconstruir a partir del orden de commits y este documento.

## Ética y honestidad

- Toda decisión técnica está respaldada por un test o por un argumento en `DECISIONS.md`.
- No hay copy-paste de modelos pre-entrenados ni de soluciones de terceros para este dataset.
- Las recomendaciones operativas (buffer P90, event uplift multipliers) están calibradas con los datos de validación reales del dataset, no con valores arbitrarios.
