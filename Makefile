# High level build commands for symforce

BUILD_DIR=build

PYTHON2=***REMOVED***/bin/mc_python
PYTHON3=***REMOVED***/bin/***REMOVED***
# TODO(hayk): Right now test_update is broken for symforce_geo_codegen_test because we don't generate
# with Python 3 because of 2/3 gen differences. Resolve this.
PYTHON=$(PYTHON3)

BLACK_EXCLUDE='symforce/codegen/python_templates|/gen/|test_data/'

CPP_FORMAT=clang-format-8

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# Build documentation, run tests, measure coverage, show in browser
all: clean docs coverage coverage_open docs_open

# Install all needed packages
all_reqs: reqs test_reqs docs_reqs

# Install python requirements for core library
reqs:
	${PYTHON} -m pip install -r requirements.txt

CPP_FILES=$(shell find . -not -path "*/lcmtypes/*" \( \
	-name "*.c" \
	-o -name "*.cpp" \
	-o -name "*.cxx" \
	-o -name "*.h" \
	-o -name "*.hpp" \
	-o -name "*.hxx" \
	-o -name "*.cu" \
	-o -name "*.cuh" \
	-o -name "*.cc" \
	-o -name "*.tcc" \
	\) )

# Format using black and clang-format
format:
	$(PYTHON) -m black --line-length 100 . --exclude $(BLACK_EXCLUDE)
	$(CPP_FORMAT) -i $(CPP_FILES)

# Check formatting using black and clang-format - print diff, do not modify files
check_format:
	$(PYTHON) -m black --line-length 100 . --exclude $(BLACK_EXCLUDE) --check --diff
	$(foreach file, $(CPP_FILES), $(CPP_FORMAT) $(file) | diff --unified $(file) - &&) true

# Check type hints using mypy
check_types:
	$(PYTHON) -m mypy --py2 . test --disallow-untyped-defs

# Lint check for formatting and type hints
# This needs pass before any merge.
lint: check_types check_format

# Clean all artifacts
clean: docs_clean coverage_clean
	rm -rf $(BUILD_DIR)

.PHONY: all reqs format check_format check_types lint clean

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

TEST_ENV=SYMFORCE_LOGLEVEL=WARNING
TEST_CMD=-m unittest discover -s test/ -p *_test.py -v

# Python files which generate code
GEN_FILES=test/*codegen*.py

test_reqs:
	${PYTHON} -m pip install -r test/requirements.txt

test_symengine:
	$(TEST_ENV) SYMFORCE_BACKEND=symengine $(PYTHON) $(TEST_CMD)

test_sympy:
	$(TEST_ENV) SYMFORCE_BACKEND=sympy $(PYTHON) $(TEST_CMD)

test_update:
	$(foreach file, $(wildcard $(GEN_FILES)), $(TEST_ENV) $(PYTHON2) $(file) --update;)

test: test_symengine test_sympy

.PHONY: test_reqs test_symengine test_sympy test

# -----------------------------------------------------------------------------
# Test coverage
# -----------------------------------------------------------------------------
COVERAGE_DIR=$(BUILD_DIR)/coverage

coverage_clean:
	rm -rf $(COVERAGE_DIR)

coverage_run:
	$(TEST_ENV) $(PYTHON) -m coverage run --source=symforce,gen --omit=symforce/codegen/python/templates/* $(TEST_CMD)

coverage_html:
	$(PYTHON) -m coverage html -d $(COVERAGE_DIR) && echo "Coverage report at $(COVERAGE_DIR)/index.html"

coverage: coverage_clean coverage_run coverage_html

coverage_open: coverage
	open $(COVERAGE_DIR)/index.html

.PHONY: coverage_clean coverage_run coverage_html coverage coverage_open

# -----------------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------------
DOCS_DIR=$(BUILD_DIR)/docs

docs_reqs:
	${PYTHON} -m pip install -r test/requirements.txt

docs_clean:
	rm -rf $(DOCS_DIR); rm -rf docs/api

docs_apidoc:
	sphinx-apidoc --separate --module-first -o docs/api ./symforce

docs_html:
	SYMFORCE_LOGLEVEL=WARNING $(PYTHON) -m sphinx -b html docs $(DOCS_DIR) -j4

docs: docs_clean docs_apidoc docs_html

docs_open: docs
	xdg-open $(DOCS_DIR)/index.html

.PHONY: docs_reqs docs_clean docs_apidoc docs_html docs docs_open

# -----------------------------------------------------------------------------
# Notebook
# -----------------------------------------------------------------------------

notebook:
	PYTHONPATH=..:../.. $(PYTHON2) -m jupyter notebook --notebook-dir=notebooks --ip=0.0.0.0 --port=8777 --no-browser

notebook_open:
	PYTHONPATH=..:../.. $(PYTHON) -m jupyter notebook --notebook-dir=notebooks --ip=localhost --port=8777
