# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Script to download the matrices used for the benchmark into this folder, in Matrix Market format.
"""
import ssgetpy  # type: ignore

# Matrix ID's used by the benchmark
matrices = [449, 1528, 2086, 1920, 1326, 664]

for matrix_id in matrices:
    ssgetpy.fetch(matrix_id, location=".")
