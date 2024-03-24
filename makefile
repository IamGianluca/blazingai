format:
	ruff format .

lint:
	ruff check --fix 
	
rebuild: compile sync install

compile:
	uv pip compile --all-extras pyproject.toml -o requirements.txt

sync:
	uv pip sync requirements.txt

install:
	uv pip install -e ".[dev,medical]"

test:
	pytest --durations=5

coverage:
	pytest --cov=blazingai

static-checks:
	mypy . --ignore-missing-imports --exclude build/ --exclude blazingai.egg-info/
