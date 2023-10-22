default: help

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: init
init: # Initialize poetry.
	@echo "Initializing poetry..."
	poetry install

	poetry config repositories.testpypi https://test.pypi.org/legacy/
	poetry config http-basic.testpypi ${TEST_PYPI_USERNAME} ${TEST_PYPI_PASSWORD}

	poetry config repositories.pypi https://upload.pypi.org/legacy/
	poetry config http-basic.pypi ${PYPI_USERNAME} ${PYPI_PASSWORD}

.PHONY: lint
lint: # Run lint with poetry.
	@echo "Running lint..."
	poetry run isort pycalf/ tests/
	poetry run black pycalf/ tests/
	poetry run pflake8 pycalf/ tests/
	poetry run mypy pycalf/ tests/

.PHONY: tests
tests: # Run lint and tests with poetry.
	make lint
	@echo "Running tests..."
	poetry run pytest --cov -v tests/

.PHONY: docs
docs: # Build documentation with poetry.
	@echo "Building documentation..."
	if [ ! -d "docs" ]; then \
		poetry run sphinx-quickstart docs --no-sep \
			-p pycalf -a konumaru -r 0.1 -l en \
			--ext-doctest \
			--ext-viewcode \
			--ext-todo \
			--ext-autodoc \
	;fi
	poetry run sphinx-apidoc -f -o docs -M pycalf \
		--ext-autodoc --ext-doctest --ext-viewcode --ext-todo
	poetry run sphinx-build -b html docs docs/build
	cd docs && make html
	