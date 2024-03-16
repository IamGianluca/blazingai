format:
	ruff format .

lint:
	ruff check --fix 
	
install:
	pip install -e .

mypy:
	mypy src/ --ignore-missing-imports

test:
	pytest -s . && \
	mypy . --ignore-missing-imports
