.PHONY: setup test lint format app exp001-app

setup:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests apps streamlit_app.py

format:
	ruff format src tests apps streamlit_app.py

app:
	PYTHONPATH=src streamlit run apps/exp001_streamlit.py

exp001-app: app
