# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

import functools

import numpy as np

import symforce.symbolic as sf
from symforce import util
from symforce.test_util import TestCase


def make_rotation_matrix() -> np.ndarray:
    # utility: gram schmidt projection
    proj = lambda v, u: ((v * u).sum() / (u * u).sum()) * u

    v = np.random.randn(3)
    w = np.random.randn(3)

    w = w - proj(w, v)
    v /= np.linalg.norm(v)
    w /= np.linalg.norm(w)

    mat = np.stack((v, w, np.cross(v, w)), axis=-1)
    return mat


class FromRotationMatrixTest(TestCase):
    def test_from_rotation_matrix(self) -> None:
        rot3_from_matrix = functools.partial(
            util.lambdify(sf.Rot3.from_rotation_matrix), epsilon=sf.numeric_epsilon
        )

        for _ in range(10000):
            R = make_rotation_matrix()
            q = rot3_from_matrix(R).to_storage()
            self.assertGreater(np.linalg.norm(q), 0.99)


if __name__ == "__main__":
    FromRotationMatrixTest.main()
