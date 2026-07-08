.PHONY: setup test lint format exp001-app

setup:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests apps

format:
	ruff format src tests apps

exp001-app:
	streamlit run apps/exp001_streamlit.py

