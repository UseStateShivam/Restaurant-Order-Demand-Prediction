.PHONY: help install download-data validate-data train evaluate test lint format run-api run-ui run-pipeline docker-build

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  install         Install project and dependencies"
	@echo "  download-data   Download raw Food Demand dataset files"
	@echo "  validate-data   Validate raw data files against schema"
	@echo "  train           Run model training pipeline"
	@echo "  evaluate        Run evaluation on trained artifacts"
	@echo "  run-pipeline    Run full end-to-end ML pipeline"
	@echo "  run-api         Start FastAPI prediction service"
	@echo "  run-ui          Start Streamlit web dashboard"
	@echo "  test            Execute pytest test suite"
	@echo "  lint            Check code quality with Ruff"
	@echo "  format          Format code with Ruff"
	@echo "  docker-build    Build serving container image"

install:
	$(PYTHON) -m pip install -e ".[dev]"

download-data:
	$(PYTHON) scripts/download_data.py

validate-data:
	$(PYTHON) -m restaurant_demand.cli validate

train:
	$(PYTHON) -m restaurant_demand.cli train

evaluate:
	$(PYTHON) -m restaurant_demand.cli evaluate

run-pipeline:
	$(PYTHON) scripts/run_pipeline.py

run-api:
	$(PYTHON) -m uvicorn restaurant_demand.api.app:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest tests -v

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests

docker-build:
	docker build -t restaurant-demand-api:latest -f docker/Dockerfile .
