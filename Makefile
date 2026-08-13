PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install build test lint format format-check doctor smoke clean

install:
	$(PIP) install -e ".[dev]" --no-build-isolation

build:
	$(PYTHON) -m build --wheel --no-isolation

test:
	$(PYTHON) -m pytest -v

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m black src tests

format-check:
	$(PYTHON) -m black --check src tests

doctor:
	$(PYTHON) -m llama_pure_compute doctor

smoke:
	$(PYTHON) -c "import torch; import llama_pure_compute; print('package:', llama_pure_compute.__version__); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('backend:', llama_pure_compute.is_cuda_backend_available())"

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .mypy_cache -prune -exec rm -rf {} +