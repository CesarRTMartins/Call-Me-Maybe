PYTHON_VERSION ?= 3.14
UV_PROJECT_ENVIRONMENT ?= $(CURDIR)/.venv
UV_PYTHON ?= $(PYTHON_VERSION)
UV_CACHE_DIR ?= ../.uv-cache
HF_HOME ?= ../.hf-cache
export UV_PROJECT_ENVIRONMENT
export UV_PYTHON
export UV_CACHE_DIR
export HF_HOME

install:
	uv sync

run:
	TRITON_DEFAULT_FP_FUSION=1 uv run python -m src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calls.json

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run flake8 --exclude=.venv,llm_sdk
	uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--explicit-package-bases \
		--exclude '^(venv|\.venv|env|llm_sdk)/'

lint-strict:
	uv run flake8 --exclude=.venv,llm_sdk
	uv run mypy . \
		--strict \
		--explicit-package-bases \
		--exclude '^(venv|\.venv|env|llm_sdk)/'

.PHONY: install run debug clean lint lint-strict