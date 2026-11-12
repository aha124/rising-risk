# Run from the repo root. `make help` lists the targets.

VENV   ?= .venv
PYTHON ?= python3.13
BIN    := $(VENV)/bin
PY     := $(BIN)/python
DBT    := $(BIN)/dbt build --project-dir warehouse --profiles-dir warehouse
SAMPLE := build/sample

.DEFAULT_GOAL := help
.PHONY: help setup data warehouse train site-data readme test parity lint format api drift site \
        sample ci docker all clean

help:  ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-11s %s\n", $$1, $$2}'

setup:  ## create .venv and install everything, at the versions in constraints.txt
	test -x $(PY) || $(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -c constraints.txt -e ".[all]"

data:  ## generate the Synthea patients into data/raw and data/sample (needs Java, about 16 minutes)
	scripts/generate_data.sh

warehouse:  ## dbt build and tests on data/raw, then export the mart and the lineage report
	$(DBT)
	$(PY) warehouse/export_marts.py
	$(PY) warehouse/export_pipeline_json.py

train:  ## fit, evaluate, explain, audit, drift, and write the site data (about 2 minutes)
	$(PY) -m riskmodel.run all
	$(PY) scripts/update_readme_metrics.py

site-data:  ## rewrite docs/data from the saved model and reports, without refitting
	$(PY) -m riskmodel.run export
	$(PY) scripts/update_readme_metrics.py

readme:  ## write the measured numbers into README.md and MODEL_CARD.md
	$(PY) scripts/update_readme_metrics.py

test: parity  ## pytest, the browser scorer parity check, and the README numbers check
	$(PY) -m pytest -q
	$(PY) scripts/update_readme_metrics.py --check

parity:  ## the in-browser scorer against Python, on 200 rows
	node docs/tests/parity.mjs

lint:  ## ruff check and format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## apply ruff fixes and formatting
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

api:  ## scoring API on http://localhost:8000 (docs at /docs)
	$(BIN)/uvicorn api.main:app --host 127.0.0.1 --port 8000

drift:  ## input and score drift for the latest index year against the training years
	$(PY) monitoring/run_drift.py \
		--reference data/marts/member_year.parquet --reference-years 2015-2021 \
		--current data/marts/member_year.parquet --current-years 2025 \
		--output build/drift/2025.json

site:  ## serve the results page on http://localhost:8080
	$(PY) -m http.server 8080 --directory docs

sample:  ## the whole pipeline on the 150-patient sample, in fast mode, written under build/
	mkdir -p $(SAMPLE)
	RR_DUCKDB_PATH=$(SAMPLE)/sample.duckdb $(DBT) --vars '{raw_path: data/sample}' \
		--target-path $(CURDIR)/$(SAMPLE)/target
	$(PY) warehouse/export_marts.py --database $(SAMPLE)/sample.duckdb --raw-path data/sample \
		--marts-dir $(SAMPLE)/marts --reports-dir $(SAMPLE)/reports
	$(PY) warehouse/export_pipeline_json.py --database $(SAMPLE)/sample.duckdb \
		--target-dir $(SAMPLE)/target --raw-path data/sample --out $(SAMPLE)/reports/pipeline.json
	RISKMODEL_FAST=1 RISKMODEL_MART_DIR=$(SAMPLE)/marts RISKMODEL_MODEL_DIR=$(SAMPLE)/model \
	RISKMODEL_REPORTS_DIR=$(SAMPLE)/reports RISKMODEL_SITE_DATA_DIR=$(SAMPLE)/site \
	RISKMODEL_MLRUNS_DIR=$(SAMPLE)/mlruns $(PY) -m riskmodel.run all
	$(PY) -m tests.model.site_schema $(SAMPLE)/site

ci: lint sample test  ## what the CI workflow runs, apart from the Docker build

docker:  ## build the API image
	docker build -f api/Dockerfile -t rising-risk-api .

all: warehouse train test lint  ## rebuild everything from data/raw and check it

clean:  ## remove build output and caches (keeps data, the model and the reports)
	rm -rf build .pytest_cache .ruff_cache warehouse/target warehouse/logs
