format:
	ruff format .

lint:
	ruff check --fix 
	
install:
	uv pip install -e ".[dev,medical]"

mypy:
	mypy src/ --ignore-missing-imports

test:
	pytest

static-checks:
	mypy . --ignore-missing-imports --exclude build/
