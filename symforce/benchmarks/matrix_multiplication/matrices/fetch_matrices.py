"""
Script to download the matrices used for the benchmark into this folder, in Matrix Market format.
"""
import ssgetpy  # type: ignore

# Matrix ID's used by the benchmark
matrices = [449, 1528, 2086, 1920, 1326, 2338, 664, 212]

for matrix_id in matrices:
    ssgetpy.fetch(matrix_id, location=".")
