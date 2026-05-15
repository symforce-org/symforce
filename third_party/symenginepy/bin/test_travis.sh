#!/usr/bin/env bash

# Exit on error
set -e
# Echo each command
set -x

python setup.py sdist
mkdir dist-extract
cd dist-extract
tar -xvf ../dist/symengine-*.tar.gz
cd symengine-*

# Build inplace so that nosetests can be run inside source directory
python3 setup.py install build_ext --inplace --symengine-dir=$our_install_dir

# Test python wrappers
python3 -m pip install pytest
python3 -m pytest -s -v $PWD/symengine/tests/test_*.py
mkdir -p empty && cd empty
python3 $PYTHON_SOURCE_DIR/bin/test_python.py
