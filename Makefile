.PHONY: install lint format test run

install:
	pip install -r requirements-dev.txt

format:
	black .
	isort .

lint:
	flake8 core/ tests/ app.py
	black --check .
	isort --check .

test:
	pytest tests/ -v --cov=core --cov-report=term-missing

run:
	streamlit run app.py
