format:
	isort . && \
	black -l 79 .

install:
	rm -rf src/ml.egg-info/ && \
	cd src && \
	pip install -e . && \
	cd ..

mypy:
	mypy src/ --ignore-missing-imports

tests:
	pytest -s src/ && \
	mypy src/ --ignore-missing-imports
