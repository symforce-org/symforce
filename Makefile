# High level build commands for symforce

BUILD_DIR=build

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# Build documentation, run tests, measure coverage, show in browser
all: clean docs coverage coverage_open docs_open

# Install all needed packages
all_reqs: reqs test_reqs docs_reqs

reqs:
	sudo pip install -r requirements.txt

format:
	black --line-length 100 . --exclude=symforce/codegen/python/templates --exclude=gen/

clean: docs_clean coverage_clean
	rm -rf $(BUILD_DIR)

.PHONY: all reqs format clean

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

TEST_ENV=SYMFORCE_LOGLEVEL=WARNING
TEST_CMD=-m unittest discover -s test/ -p *_test.py -v

test_reqs:
	sudo pip install -r test/requirements.txt

test:
	$(TEST_ENV) python $(TEST_CMD)

.PHONY: test_reqs test

# -----------------------------------------------------------------------------
# Test coverage
# -----------------------------------------------------------------------------
COVERAGE_DIR=$(BUILD_DIR)/coverage

coverage_clean:
	rm -rf $(COVERAGE_DIR)

coverage_run:
	$(TEST_ENV) coverage run --source=symforce --omit=symforce/codegen/python/templates/* $(TEST_CMD)

coverage_html:
	coverage html -d $(COVERAGE_DIR)

coverage: coverage_clean coverage_run coverage_html

coverage_open: coverage
	open $(COVERAGE_DIR)/index.html

.PHONY: coverage_clean coverage_run coverage_html coverage coverage_open

# -----------------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------------
DOCS_DIR=$(BUILD_DIR)/docs

docs_reqs:
	sudo pip install -r test/requirements.txt

docs_clean:
	rm -rf $(DOCS_DIR); rm -rf docs/api

docs_apidoc:
	sphinx-apidoc --separate --module-first -o docs/api ./symforce

docs_html:
	SYMFORCE_LOGLEVEL=WARNING sphinx-build -b html docs $(DOCS_DIR) -j4

docs: docs_clean docs_apidoc docs_html

docs_open: docs
	open $(DOCS_DIR)/index.html

.PHONY: docs_reqs docs_clean docs_apidoc docs_html docs docs_open

# -----------------------------------------------------------------------------
# Notebook
# -----------------------------------------------------------------------------

notebook:
	PYTHONPATH=..:../.. jupyter notebook --notebook-dir=docs/notebooks --ip=localhost --port=8777
