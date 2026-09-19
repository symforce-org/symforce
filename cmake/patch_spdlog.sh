#!/bin/bash

# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

set -eu

spdlog_src="$1"
shift

# FetchContent re-runs the patch step on trees that may already be patched. The series overlaps
# itself, so a per-patch check can't tell "already applied" from "does not apply"; stamp the whole
# series instead.
stamp="${spdlog_src}/.symforce_patches_applied"
series="$(cat "$@" | sha256sum | cut -d' ' -f1)"

if [ -f "${stamp}" ] && [ "$(cat "${stamp}")" = "${series}" ]; then
  exit 0
fi

for patch_file in "$@"; do
  patch -p1 -d "${spdlog_src}" -i "${patch_file}"
done

echo "${series}" > "${stamp}"
