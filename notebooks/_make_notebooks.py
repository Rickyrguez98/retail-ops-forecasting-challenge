"""Generate the four analysis notebooks programmatically.

This script writes the notebooks as JSON. It is intentionally lightweight —
the goal is reproducibility: regenerate notebooks from a single source file
rather than commit binary outputs that drift.

After running, execute the notebooks with:
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent


def _md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(text)


def _code(src: str) -> dict:
    return nbf.v4.new_code_cell(src)


def _save(name: str, cells: list[dict]) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    p = NB_DIR / name
    with open(p, "w") as f:
        nbf.write(nb, f)
    print("wrote", p)


PREAMBLE = """\
import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from retail_ops_forecasting.config import load_config
from retail_ops_forecasting.data import load_transactions, load_stores, load_calendar, build_merged
sns.set_style('whitegrid')
cfg = load_config(ROOT / 'configs' / 'config.yaml')
"""


def nb_01_initial_eda():
    cells = [
        _md("# 01 — Initial EDA\n\nEntender el negocio: estructura del dataset, missingness, patrones temporales por tienda/categoría/región.\n\n"
            "Periodo: 2023-01-01 → 2024-02-29. 80 tiendas × 6 categorías × 425 días."),
        _code(PREAMBLE),
        _md("## Carga y shape"),
        _code("tx = load_transactions(cfg.paths.raw_dir / cfg.data.transactions_file)\n"
              "st = load_stores(cfg.paths.raw_dir / cfg.data.stores_file)\n"
              "cal = load_calendar(cfg.paths.raw_dir / cfg.data.calendar_file)\n"
              "print('transactions:', tx.shape, '| stores:', st.shape, '| calendar:', cal.shape)\n"
              "tx.head(3)"),
        _md("## Cobertura: tiendas × categorías × días"),
        _code("print('stores:', tx.store_id.nunique(), '| categories:', tx.category.nunique(), '| days:', tx.date.nunique())\n"
              "rows_expected = tx.store_id.nunique() * tx.category.nunique() * tx.date.nunique()\n"
              "print(f'rows expected = {rows_expected:,}; observed = {len(tx):,}; ratio = {len(tx)/rows_expected:.4f}')"),
        _md("## Missingness por columna"),
        _code("miss = tx.isna().sum().sort_values(ascending=False)\n"
              "miss = miss[miss > 0]\n"
              "(miss / len(tx) * 100).round(3).to_frame('pct_missing')"),
        _md("Las columnas con datos faltantes son consistentes con el diccionario: `cash_transactions` y `amount_cash` (~5.9%) por fallas de POS; `units_sold` (3%); `avg_ticket` (~2.4%); `replenishment_signal` (0.47% — sólo últimos días)."),
        _md("## Distribución de `total_transactions`"),
        _code("fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
              "ax[0].hist(tx['total_transactions'], bins=80, color='steelblue')\n"
              "ax[0].set_title('total_transactions (raw)')\n"
              "ax[0].set_xlabel('transactions/day')\n"
              "ax[1].hist(np.log1p(tx['total_transactions']), bins=80, color='steelblue')\n"
              "ax[1].set_title('log1p(total_transactions)')\n"
              "plt.tight_layout(); plt.savefig(ROOT / 'reports' / 'figures' / 'eda_demand_dist.png', dpi=120); plt.show()"),
        _md("## Estacionalidad semanal por formato"),
        _code("df = tx.merge(st, on='store_id').merge(cal, on='date')\n"
              "agg = df.groupby(['store_format', 'day_of_week'])['total_transactions'].mean().reset_index()\n"
              "fig, ax = plt.subplots(figsize=(9, 4))\n"
              "for fmt, sub in agg.groupby('store_format'):\n"
              "    ax.plot(sub['day_of_week'], sub['total_transactions'], marker='o', label=fmt)\n"
              "ax.set_xticks(range(7)); ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])\n"
              "ax.set_ylabel('avg total_transactions'); ax.set_title('Demanda promedio por día de la semana × formato'); ax.legend()\n"
              "plt.tight_layout(); plt.savefig(ROOT / 'reports' / 'figures' / 'eda_dow_format.png', dpi=120); plt.show()"),
        _md("Los Supercenters muestran el patrón semanal más fuerte (pico fin de semana). Express es más plano."),
        _md("## Demanda diaria total — vista temporal"),
        _code("daily = df.groupby('date')['total_transactions'].sum()\n"
              "fig, ax = plt.subplots(figsize=(12, 4))\n"
              "ax.plot(daily.index, daily.values)\n"
              "for col, label, color in [('is_buen_fin','Buen Fin','red'), ('is_navidad_season','Navidad','green'), ('is_semana_santa','Semana Santa','purple')]:\n"
              "    evt = cal[cal[col].fillna(False).astype(bool)]['date']\n"
              "    if len(evt):\n"
              "        ax.axvspan(evt.min(), evt.max(), color=color, alpha=0.10, label=label)\n"
              "ax.set_title('Demanda diaria total (todas las tiendas y categorías)'); ax.legend()\n"
              "plt.tight_layout(); plt.savefig(ROOT / 'reports' / 'figures' / 'eda_daily_total.png', dpi=120); plt.show()"),
        _md("## Resumen por categoría"),
        _code("cat_summary = tx.groupby('category').agg(\n"
              "    rows=('total_transactions','size'),\n"
              "    mean_tx=('total_transactions','mean'),\n"
              "    median_tx=('total_transactions','median'),\n"
              "    mean_units=('units_sold','mean'),\n"
              "    cash_share=('amount_cash','mean'),\n"
              ").round(2)\n"
              "cat_summary"),
        _md("## Notas para feature engineering\n\n"
            "1. Series suficientemente largas (425 días) para lags 1/7/14/28.\n"
            "2. Estacionalidad semanal clara → lag-7 será un baseline fuerte.\n"
            "3. Eventos (Buen Fin, Navidad) visibles → calendar flags como features.\n"
            "4. Distribución de demanda con cola larga → considerar log o usar HGB (robusto a no-normalidad)."),
    ]
    _save("01_initial_eda.ipynb", cells)


def nb_02_statistical_analysis():
    cells = [
        _md("# 02 — Statistical analysis\n\nValidamos hipótesis con tests formales y cuantificamos el impacto de eventos y segmentos."),
        _code(PREAMBLE),
        _code("from scipy import stats\n"
              "df = build_merged(cfg)\n"
              "df.shape"),
        _md("## ANOVA — `total_transactions` por formato"),
        _code("df['total_transactions_f'] = df['total_transactions'].astype('float64')\n"
              "groups = [g['total_transactions_f'].dropna().values for _, g in df.groupby('store_format')]\n"
              "f, p = stats.f_oneway(*groups)\n"
              "print(f'F={f:.2f}, p={p:.3e}')"),
        _md("F muy grande, p ≈ 0 — los formatos difieren significativamente. (Confirma intuición; el dataset no es uniforme.)"),
        _md("## ANOVA — `total_transactions` por región"),
        _code("groups = [g['total_transactions_f'].dropna().values for _, g in df.groupby('region')]\n"
              "f, p = stats.f_oneway(*groups)\n"
              "print(f'F={f:.2f}, p={p:.3e}')"),
        _md("## Efecto de eventos (welch t-test, regular vs evento)"),
        _code("def welch(evt_col):\n"
              "    a = df.loc[df[evt_col].fillna(False).astype(bool), 'total_transactions_f'].dropna()\n"
              "    b = df.loc[~df[evt_col].fillna(False).astype(bool), 'total_transactions_f'].dropna()\n"
              "    t, p = stats.ttest_ind(a, b, equal_var=False)\n"
              "    return {'event': evt_col, 'mean_event': float(a.mean()), 'mean_regular': float(b.mean()), 'uplift_pct': 100*(float(a.mean())/float(b.mean())-1), 't': float(t), 'p': float(p)}\n"
              "import pandas as pd\n"
              "pd.DataFrame([welch(c) for c in ['is_buen_fin','is_semana_santa','is_navidad_season','is_payday','is_weekend','is_holiday']]).round(3)"),
        _md("Los uplifts y los t-stats cuantifican el efecto de cada evento. Buen Fin y temporada navideña muestran efectos claros; quincena (`is_payday`) también es relevante."),
        _md("## Correlación de `replenishment_signal` con same-day target"),
        _code("sub = df.dropna(subset=['replenishment_signal','total_transactions'])\n"
              "a = sub['replenishment_signal'].astype(float).values\n"
              "b = sub['total_transactions'].astype(float).values\n"
              "corr = float(((a-a.mean())*(b-b.mean())).sum() / (((a-a.mean())**2).sum()**0.5 * ((b-b.mean())**2).sum()**0.5))\n"
              "print(f'corr = {corr:.3f}  (alta → señal sospechosa, excluida del modelo)')"),
        _md("## Cash share por formato y región"),
        _code("agg = df.assign(cash_share=df['amount_cash']/df['amount_total']).dropna(subset=['cash_share'])\n"
              "agg.groupby(['store_format','region'])['cash_share'].mean().unstack().round(3)"),
        _md("La proporción de cash difiere por formato y región; alimentará features de cash_share lagged."),
    ]
    _save("02_statistical_analysis.ipynb", cells)


def nb_03_model_experiments():
    cells = [
        _md("# 03 — Model experiments\n\nComparamos baselines vs. HGB. El código vive en `src/`; aquí lo importamos y narramos resultados."),
        _code(PREAMBLE + "\nfrom retail_ops_forecasting import baselines, modeling, features, splits, metrics\n"
              "from retail_ops_forecasting.config import load_model_config\n"
              "mcfg = load_model_config(ROOT / 'configs' / 'model_config.yaml')"),
        _md("## Cargar feature matrix de demand"),
        _code("p = ROOT / 'data' / 'processed' / 'demand_features.csv'\n"
              "if (p.with_suffix('.parquet')).exists():\n"
              "    feat = pd.read_parquet(p.with_suffix('.parquet'))\n"
              "else:\n"
              "    feat = pd.read_csv(p, parse_dates=['date'])\n"
              "feat.shape"),
        _md("## Split temporal"),
        _code("s = splits.time_based_split(feat, cfg.splits)\n"
              "print('train:', len(s.train), 'val:', len(s.val), 'test:', len(s.test))"),
        _md("## Baseline lag-7 — métricas val/test"),
        _code("target = cfg.targets.demand_primary\n"
              "feat['lag7'] = baselines.naive_lag(feat, target, lag=7, entity_cols=['store_id','category'])\n"
              "for name, idx in [('val', s.val), ('test', s.test)]:\n"
              "    sub = feat.loc[idx].dropna(subset=[target,'lag7'])\n"
              "    m = metrics.summarize(sub[target], sub['lag7'])\n"
              "    print(name, {k: round(v,4) if isinstance(v,float) else v for k,v in m.items()})"),
        _md("## Cargar resultados experimentales (después de `make train-demand`)"),
        _code("import json\n"
              "p = ROOT / 'reports' / 'demand_experiment_summary.json'\n"
              "if p.exists():\n"
              "    sm = json.loads(p.read_text())\n"
              "    rows = []\n"
              "    for run, sp in sm.items():\n"
              "        for split, m in sp.items():\n"
              "            row = {'run': run, 'split': split, **{k:v for k,v in m.items() if isinstance(v,(int,float))}}\n"
              "            rows.append(row)\n"
              "    pd.DataFrame(rows).round(4)\n"
              "else:\n"
              "    print('Run `make train-demand` first.')"),
    ]
    _save("03_model_experiments.ipynb", cells)


def nb_04_error_analysis():
    cells = [
        _md("# 04 — Error analysis\n\nDónde el modelo gana, dónde pierde. Lectura por segmento (formato, región, categoría) y por evento (Buen Fin, quincena, fin de semana)."),
        _code(PREAMBLE),
        _md("## Cargar predicciones de test"),
        _code("from retail_ops_forecasting.data import load_stores, load_calendar\n"
              "demand_test = pd.read_csv(ROOT / 'data' / 'processed' / 'demand_predictions_test.csv', parse_dates=['date'])\n"
              "cash_test = pd.read_csv(ROOT / 'data' / 'processed' / 'cash_predictions_test.csv', parse_dates=['date'])\n"
              "stores = load_stores(cfg.paths.raw_dir / cfg.data.stores_file)\n"
              "cal = load_calendar(cfg.paths.raw_dir / cfg.data.calendar_file)\n"
              "demand_test.head(2)"),
        _md("## Residuos por categoría"),
        _code("d = demand_test.rename(columns={cfg.targets.demand_primary:'y_true'}).merge(stores, on='store_id', how='left')\n"
              "d['residual'] = d['y_true'] - d['y_pred']\n"
              "fig, ax = plt.subplots(figsize=(9, 4))\n"
              "sns.boxplot(data=d, x='category', y='residual', ax=ax)\n"
              "ax.axhline(0, color='k', lw=0.7); ax.set_title('Residuos demand por categoría')\n"
              "plt.tight_layout(); plt.savefig(ROOT / 'reports' / 'figures' / 'error_demand_by_category.png', dpi=120); plt.show()"),
        _md("## WAPE por evento"),
        _code("from retail_ops_forecasting.metrics import wape\n"
              "d = d.merge(cal[['date','is_buen_fin','is_navidad_season','is_payday','is_weekend','is_holiday','is_semana_santa']], on='date', how='left')\n"
              "d['event'] = 'regular'\n"
              "for col, lab in [('is_buen_fin','buen_fin'),('is_semana_santa','semana_santa'),('is_navidad_season','navidad'),('is_payday','payday'),('is_holiday','holiday'),('is_weekend','weekend')]:\n"
              "    d.loc[d[col].fillna(False).astype(bool), 'event'] = lab\n"
              "rows = [{'event': e, 'wape': wape(g['y_true'], g['y_pred']), 'n': len(g)} for e, g in d.groupby('event')]\n"
              "pd.DataFrame(rows).round(4)"),
        _md("## Cash — coverage por tienda (P90)"),
        _code("recs = pd.read_csv(ROOT / 'reports' / 'cash_forecast_recommendations.csv', parse_dates=['date'])\n"
              "tcol = cfg.targets.cash_primary\n"
              "recs['ok'] = recs['recommended_cash'] >= recs[tcol]\n"
              "cov = recs.groupby('store_id')['ok'].mean().sort_values()\n"
              "print(f'coverage por tienda — min {cov.min():.2f}, median {cov.median():.2f}, max {cov.max():.2f}')\n"
              "fig, ax = plt.subplots(figsize=(9,3))\n"
              "ax.hist(cov, bins=20, color='seagreen', alpha=0.8); ax.axvline(0.9, color='k', linestyle='--')\n"
              "ax.set_title('Coverage por tienda (P90 buffer)')\n"
              "plt.tight_layout(); plt.savefig(ROOT / 'reports' / 'figures' / 'cash_coverage_per_store.png', dpi=120); plt.show()"),
    ]
    _save("04_error_analysis.ipynb", cells)


if __name__ == "__main__":
    nb_01_initial_eda()
    nb_02_statistical_analysis()
    nb_03_model_experiments()
    nb_04_error_analysis()
