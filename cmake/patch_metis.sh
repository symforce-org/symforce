#!/bin/bash

set -ex

metis_src="$1"

# They add a "strings" executable in the test directory and have no option to not build the test
# directory, so we need to disable that.
sed -i".bak" -e "s/add_subdirectory(\"test\")//" "${metis_src}/GKlib/CMakeLists.txt"
rm "${metis_src}/GKlib/CMakeLists.txt.bak"
