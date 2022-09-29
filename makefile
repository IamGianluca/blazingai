format:
	usort format . && \
	black -l 79 .

install:
	pip install -e .

mypy:
	mypy src/ --ignore-missing-imports

test:
	pytest -s . && \
	mypy . --ignore-missing-imports
