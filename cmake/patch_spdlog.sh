#!/bin/bash

# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

set -eu

spdlog_src="$1"
shift

# Apply the patches that make spdlog 1.9.2 compile against fmt 12. FetchContent re-runs the patch
# step on trees that may already be patched, so skip any patch whose changes are already present -- a
# successful reverse dry-run means the tree contains that patch.
for patch_file in "$@"; do
  if patch -p1 -R -s --dry-run -d "${spdlog_src}" -i "${patch_file}" >/dev/null 2>&1; then
    continue
  fi

  patch -p1 -d "${spdlog_src}" -i "${patch_file}"
done
