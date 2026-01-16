.PHONY: install test coverage train serve ge_init ge_validate

install:
	pip install -r requirements.txt

test:
	pytest -q

coverage:
	pytest --cov=src --cov-report=term-missing -q

train:
	python -m src.train

serve:
	uvicorn src.serve:app --reload --port 8000

ge_init:
	python -m src.data ge-init

ge_validate:
	python -m src.data ge-validate
