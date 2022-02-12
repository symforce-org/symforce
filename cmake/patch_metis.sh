#!/bin/bash

# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

set -ex

metis_src="$1"

# They add a "strings" executable in the test directory and have no option to not build the test
# directory, so we need to disable that.
sed -i".bak" -e "s/add_subdirectory(\"test\")//" "${metis_src}/GKlib/CMakeLists.txt"
rm "${metis_src}/GKlib/CMakeLists.txt.bak"
