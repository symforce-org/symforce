import numpy as np

import symforce
from symforce import geo
from symforce import sympy as sm
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoMatrixTest(LieGroupOpsTestMixin, TestCase):
    """
    Test the Matrix geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls):
        return geo.Matrix([-0.2, 5.3, 1.2])

    # TODO(hayk): Test from_storage for matrices - how should shape be preserved?

    def test_matrix_initialization(self):
        # type: () -> None
        """
        Tests:
            Matrix.zero
            Matrix.zeros
            Matrix.one
            Matrix.ones
            Matrix.diag
            Matrix.eye
            Matrix.matrix_identity
            Matrix.column_stack
        """
        element = GeoMatrixTest.element()

        dims = element.MATRIX_DIMS
        self.assertEqual(element.zero(), geo.Matrix.zeros(dims[0], dims[1]))
        self.assertEqual(element.one(), geo.Matrix.ones(dims[0], dims[1]))
        self.assertNotEqual(element.one(), element.zero())

        eye_matrix = geo.Matrix.diag([1, 1, 1])
        self.assertEqual(eye_matrix, geo.Matrix.eye(3))
        self.assertEqual(eye_matrix, eye_matrix.matrix_identity())

        nonsquare_diag_matrix = geo.Matrix([[1, 0, 0], [0, 1, 0]])
        self.assertEqual(nonsquare_diag_matrix, geo.Matrix.eye(2, 3))

        vec1 = geo.V3([1, 2, 3])
        vec2 = geo.V3([4, 5, 6])
        vec3 = geo.V3([7, 8, 9])
        mat = geo.Matrix([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        self.assertEqual(mat, geo.Matrix.column_stack(vec1, vec2, vec3))
        self.assertEqual(geo.Matrix(), geo.Matrix.column_stack())

    def test_matrix_operations(self):
        """
        Tests:
            Matrix.Matrix_inverse
            Matrix.__add__ (scalar_like)
            Matrix.__div__ (scalar_like)
        """

        test_matrix = geo.Matrix([[2, 0], [0, 4]])
        inv_matrix = test_matrix.matrix_inverse()
        self.assertEqual(geo.Matrix.eye(2), test_matrix * inv_matrix)

        self.assertEqual(geo.Matrix.ones(2, 2), geo.Matrix.zeros(2, 2) + 1)

        diag_matrix = 2 * geo.Matrix.eye(2)
        self.assertEqual(geo.Matrix.eye(2), diag_matrix / 2)

        self.assertEqual(geo.Matrix.eye(2), test_matrix / test_matrix)

    def test_symbolic_operations(self):
        """
        Tests:
            Matrix.symbolic
            Matrix.simplify
            Matrix.evalf
            Matrix.to_numpy
        """

        sym_vector = geo.V3().symbolic("vector")
        self.assertEqual(sym_vector.MATRIX_DIMS, (3, 1))
        sym_matrix = geo.I3().symbolic("matrix")
        self.assertEqual(sym_matrix.MATRIX_DIMS, (3, 3))

        x = sm.Symbol("x")
        unsimple_matrix = geo.Matrix([x ** 2 - x, 0])
        simple_matrix = geo.Matrix([x * (x - 1), 0])
        self.assertEqual(unsimple_matrix.simplify(), simple_matrix)

        pi_mat = sm.pi * geo.I2()
        pi_mat_num = geo.Matrix([[3.14159265, 0], [0, 3.14159265]])
        self.assertNear(pi_mat.evalf().to_storage(), pi_mat_num.to_storage())

        numpy_mat = np.array([[2.0, 1.0], [4.0, 3.0]])
        geo_mat = geo.Matrix([[2.0, 1.0], [4.0, 3.0]])
        self.assertNear(numpy_mat, geo_mat.to_numpy())

    def test_constructor_helpers(self):
        """
        Tests:
            VectorX, ZX, IX constructor helpers
        """
        vector_constructors = [
            geo.V1,
            geo.V2,
            geo.V3,
            geo.V4,
            geo.V5,
            geo.V6,
            geo.V7,
            geo.V8,
            geo.V9,
        ]
        for i, vec in enumerate(vector_constructors):
            self.assertEqual(vec(), geo.Matrix.zeros(i + 1, 1))
            rand_vec = np.random.rand(i + 1)
            self.assertEqual(vec(rand_vec), geo.Matrix(rand_vec))
            self.assertEqual(vec(*rand_vec), geo.Matrix(rand_vec))

            rand_vec_long = np.random.rand(i + 2)
            self.assertRaises(ArithmeticError, vec, rand_vec_long)
            self.assertRaises(ArithmeticError, vec, *rand_vec_long)

        zero_matrix_constructors = [
            geo.Z1,
            geo.Z2,
            geo.Z3,
            geo.Z4,
            geo.Z5,
            geo.Z6,
            geo.Z7,
            geo.Z8,
            geo.Z9,
        ]
        for i, mat in enumerate(zero_matrix_constructors):
            self.assertEqual(mat(), geo.Matrix.zeros(i + 1, 1))

        zero_matrix_constructors = [
            geo.Z11,
            geo.Z22,
            geo.Z33,
            geo.Z44,
            geo.Z55,
            geo.Z66,
            geo.Z77,
            geo.Z88,
            geo.Z99,
        ]
        for i, mat in enumerate(zero_matrix_constructors):
            self.assertEqual(mat(), geo.Matrix.zeros(i + 1, i + 1))

        eye_matrix_constructors = [
            geo.I1,
            geo.I2,
            geo.I3,
            geo.I4,
            geo.I5,
            geo.I6,
            geo.I7,
            geo.I8,
            geo.I9,
        ]
        for i, mat in enumerate(eye_matrix_constructors):
            self.assertEqual(mat(), geo.Matrix.eye(i + 1))


if __name__ == "__main__":
    np.random.seed(42)
    TestCase.main()
