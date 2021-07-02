#!/bin/bash

set -ex

cd symengine/utilities/matchpycpp
python generate_tests.py
cd ../../..
git diff --exit-code
