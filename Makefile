.PHONY: test train eval search docs lint

test:
	python -m pytest -q -m "not gpu and not ray"

train:
	python scripts/train.py --config configs/methods/grpo.yaml

eval:
	python scripts/evaluate.py --help

search:
	python scripts/run_search.py --config configs/search/pgts.yaml --output experiments/search_out.jsonl

lint:
	ruff check src tests scripts

docs:
	cd docs && sphinx-build -b html . _build/html
