format:
	ruff format .

lint:
	ruff check --fix 
	
install:
	uv pip install -e ".[dev,medical]"

mypy:
	mypy src/ --ignore-missing-imports

test:
	pytest -s . && \
	mypy . --ignore-missing-imports --exclude build/
