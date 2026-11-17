.PHONY: install install-dev data-check clean-data features train score export-bi pipeline notebooks test lint format serve docker-build docker-run

PY ?= python
DAY ?= 45

install:            ## runtime dependencies (pipeline + API)
	$(PY) -m pip install -c requirements.lock -e .

install-dev:        ## everything: analysis, notebooks, tests, lint
	$(PY) -m pip install -c requirements.lock -e ".[analysis,dev]"

data-check:
	@test -f data/raw/challenge_edMachina.csv || (echo "Missing data/raw/challenge_edMachina.csv - see data/README.md" && exit 1)

clean-data: data-check
	early-warning clean

features:
	early-warning features

train:
	early-warning train

score:
	early-warning score --day $(DAY)

export-bi:
	early-warning export-bi

pipeline: data-check  ## clean -> features -> train -> score -> export-bi
	early-warning all

notebooks:          ## re-execute the notebooks in place
	cd notebooks && for nb in 01_eda 02_statistical_inference 03_modeling_results; do \
		jupyter nbconvert --to notebook --execute --inplace $$nb.ipynb; done

test:
	$(PY) -m pytest --cov=early_warning --cov-report=term-missing

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

serve:
	uvicorn early_warning.api.app:app --host 0.0.0.0 --port 8000 --workers 2

docker-build:
	docker build -t early-warning-api:latest .

docker-run:
	docker run --rm -p 8000:8000 early-warning-api:latest
