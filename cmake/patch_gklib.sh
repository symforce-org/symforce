#!/bin/bash

# This script might be run multiple times on the same source, so it should be idempotent

set -ex

gklib_src="$1"

# GKlib adds -march=native on gcc, which is incorrect if cross-compiling.  This cmake file is
# also included directly in METIS, so we can't just overwrite GKlib_COPTIONS, we have to patch.
sed -i".bak" -e "s/-march=native//" "${gklib_src}/GKlibSystem.cmake"
rm "${gklib_src}/GKlibSystem.cmake.bak"

# They also add a "strings" executable in the test directory and have no option to not build
# the test directory, so we need to disable that.
sed -i".bak" -e "s/add_subdirectory(\"test\")//" "${gklib_src}/CMakeLists.txt"
rm "${gklib_src}/CMakeLists.txt.bak"
