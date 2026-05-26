.PHONY: help install quality prepare train-demand train-demand-lgb train-cash evaluate report all test lint clean

PYTHON ?= python3
N_TRIALS ?= 30

help:
	@echo "Targets:"
	@echo "  install            Install pinned dependencies"
	@echo "  quality            Run data quality checks"
	@echo "  prepare            Build interim + processed datasets"
	@echo "  train-demand       Train demand baselines + HGB + event uplift"
	@echo "  train-demand-lgb   Train tuned LightGBM benchmark (Optuna Bayesian opt, N_TRIALS=30)"
	@echo "  train-cash         Train cash forecasting model + buffer rule"
	@echo "  evaluate           Generate test-set metrics + segment breakdowns"
	@echo "  report             Generate reports/figures and summary tables"
	@echo "  all                quality -> prepare -> train-demand -> train-demand-lgb -> train-cash -> evaluate -> report"
	@echo "  test               Run pytest suite"
	@echo "  lint               Run black/isort/ruff in check mode"
	@echo "  clean              Remove generated artifacts (keeps raw data + reports)"

install:
	$(PYTHON) -m pip install -r requirements.txt

quality:
	$(PYTHON) scripts/run_quality_checks.py

prepare:
	$(PYTHON) scripts/prepare_data.py

train-demand:
	$(PYTHON) scripts/train_demand_model.py

train-demand-lgb:
	$(PYTHON) scripts/train_demand_lightgbm.py --n-trials $(N_TRIALS)

train-cash:
	$(PYTHON) scripts/train_cash_model.py

evaluate:
	$(PYTHON) scripts/evaluate_models.py

report:
	$(PYTHON) scripts/generate_report.py

all: quality prepare train-demand train-demand-lgb train-cash evaluate report

test:
	$(PYTHON) -m pytest -ra

lint:
	@command -v black >/dev/null && black --check src tests scripts || echo "black not installed — skipping"
	@command -v isort >/dev/null && isort --check src tests scripts || echo "isort not installed — skipping"
	@command -v ruff  >/dev/null && ruff check src tests scripts   || echo "ruff not installed — skipping"

clean:
	rm -rf data/interim/* data/processed/* models/* mlruns/* reports/figures/*.png
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
