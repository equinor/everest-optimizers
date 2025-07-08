# Makefile for everest-optimizers

# Use python3 by default. User can override it. e.g., make PYTHON=python3.11
PYTHON ?= python3
VENV_DIR = .venv

# Phony targets don't represent files
.PHONY: all help clean install test

all: help

help:
	@echo "Available commands:"
	@echo "  make install  - Create a virtual environment and install dependencies"
	@echo "  make test     - Run the test suite"
	@echo "  make clean    - Remove virtual environment and build artifacts"

# Create virtual environment if it doesn't exist
$(VENV_DIR)/bin/activate:
	$(PYTHON) -m venv $(VENV_DIR)

install: $(VENV_DIR)/bin/activate
	@echo "Installing dependencies..."
	. $(VENV_DIR)/bin/activate; \
	python -m pip install --upgrade pip; \
	python -m pip install -e ".[test,dev]"

test:
	@echo "Running tests..."
	. $(VENV_DIR)/bin/activate; \
	pytest tests/

clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV_DIR) build/ dist/ *.egg-info src/everest_optimizers.egg-info dakota-packages/OPTPP/build
	@find . -type d -name "__pycache__" -exec rm -r {} +
