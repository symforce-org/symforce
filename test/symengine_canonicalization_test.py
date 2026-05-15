# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Canonicalization tests: different-but-equivalent constructions should produce
identical symbolic objects and identical string representations.
"""

import symforce.symbolic as sf
from symforce.test_util import TestCase
from symforce.test_util import symengine_only


class SymengineCanonicalizationTest(TestCase):
    """Verify that equivalent expression constructions canonicalize identically."""

    @symengine_only
    def test_addition_commutativity(self) -> None:
        """x + y and y + x should canonicalize to the same expression."""
        x, y = sf.symbols("x y")
        self.assertEqual(x + y, y + x)
        self.assertEqual(str(x + y), str(y + x))

    @symengine_only
    def test_doubling_canonicalizes_to_multiplication(self) -> None:
        """2*x and x + x should produce the same canonical form."""
        x = sf.Symbol("x")
        self.assertEqual(2 * x, x + x)
        self.assertEqual(str(2 * x), str(x + x))

    @symengine_only
    def test_squaring_canonicalizes_to_power(self) -> None:
        """x**2 and x*x should produce the same canonical form."""
        x = sf.Symbol("x")
        self.assertEqual(x**2, x * x)
        self.assertEqual(str(x**2), str(x * x))

    @symengine_only
    def test_trig_pythagorean_identity(self) -> None:
        """simplify(sin(x)**2 + cos(x)**2) should equal 1."""
        x = sf.Symbol("x")
        simplified = sf.simplify(sf.sin(x) ** 2 + sf.cos(x) ** 2)
        self.assertEqual(simplified, sf.S.One)

    @symengine_only
    def test_rotation_matrix_construction(self) -> None:
        """Rot3.from_angle_axis around z should match the canonical z-axis rotation matrix."""
        theta = sf.Symbol("theta")
        rot_a = sf.Rot3.from_angle_axis(angle=theta, axis=sf.V3(0, 0, 1))
        mat_a = rot_a.to_rotation_matrix()
        mat_b = sf.Matrix(
            [
                [sf.cos(theta), -sf.sin(theta), 0],
                [sf.sin(theta), sf.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        for row in range(3):
            for col in range(3):
                diff = sf.simplify(mat_a[row, col] - mat_b[row, col])
                self.assertEqual(diff, 0, f"Element [{row},{col}] does not match")


if __name__ == "__main__":
    TestCase.main()
