# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce.symbolic as sf
from symforce import typing as T
from symforce.values import Values


def load(filename: T.Openable) -> tuple[list[int], list[tuple[int, int]], Values]:
    """
    Load a g2o file and return the pose IDs, edges (pairs of pose IDs) and values (initial poses,
    relative pose measurements and their sqrt information matrices)
    """

    def to_float(l: T.Sequence[str]) -> list[float]:
        return [float(x) for x in l]

    def pose_from_g2o(l: T.Sequence[str]) -> sf.Pose3:
        storage_g2o = to_float(l)
        return sf.Pose3(R=sf.Rot3.from_storage(storage_g2o[3:]), t=sf.V3(storage_g2o[:3]))

    ids = []
    edges = []
    values = Values()

    with open(filename) as f:
        for line_str in f:
            line = line_str.split()
            if line[0] == "VERTEX_SE3:QUAT":
                i = int(line[1])
                ids.append(i)
                values[f"p_{i}"] = pose_from_g2o(line[2:])
            elif line[0] == "EDGE_SE3:QUAT":
                i, j = int(line[1]), int(line[2])
                edges.append((i, j))
                values[f"e_{i}_{j}"] = pose_from_g2o(line[3:10])

                # Check frame
                values[f"i_{i}_{j}"] = info_storage_to_sqrt_info(to_float(line[10:]))

    return ids, edges, values.to_numerical()


def info_storage_to_sqrt_info(info_storage: T.Sequence[float]) -> np.ndarray:
    info = np.zeros((6, 6))
    ix = 0
    for i in range(info.shape[0]):
        info[i, i:] = info_storage[ix : ix + (6 - i)]
        info[i:, i] = info_storage[ix : ix + (6 - i)]
        ix += 6 - i

    info = info[(3, 4, 5, 0, 1, 2), :]
    info = info[:, (3, 4, 5, 0, 1, 2)]

    # Convert the information matrix to a square root information matrix
    sqrt_info = np.linalg.cholesky(info).T

    return sqrt_info
