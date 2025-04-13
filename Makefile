default: help

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: init
init: # Initialize virtual environment with uv.
	@echo "Initializing virtual environment with uv..."
	python -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install uv
	.venv/bin/uv pip install -e .

.PHONY: lint
lint: # Run lint.
	@echo "Running lint..."
	.venv/bin/isort pycalf/ tests/
	.venv/bin/black pycalf/ tests/
	.venv/bin/pflake8 pycalf/ tests/
	.venv/bin/mypy pycalf/ tests/

.PHONY: tests
tests: # Run lint and tests.
	make lint
	@echo "Running tests..."
	.venv/bin/pytest --cov -v tests/

.PHONY: docs
docs: # Build documentation.
	@echo "Building documentation..."
	if [ ! -d "docs" ]; then \
		.venv/bin/sphinx-quickstart docs --no-sep \
			-p pycalf -a konumaru -r 0.1 -l en \
			--ext-doctest \
			--ext-viewcode \
			--ext-todo \
			--ext-autodoc \
	;fi
	.venv/bin/sphinx-apidoc -f -o docs -M pycalf \
		--ext-autodoc --ext-doctest --ext-viewcode --ext-todo
	.venv/bin/sphinx-build -b html docs docs/build
	cd docs && make html

.PHONY: publish
publish: # Publish package to PyPI.
	@echo "Building package..."
	.venv/bin/pip install build twine
	.venv/bin/python -m build
	@echo "Publishing package to Test PyPI..."
	.venv/bin/twine upload --repository testpypi dist/*

	