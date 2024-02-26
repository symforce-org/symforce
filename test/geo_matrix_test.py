# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce
import symforce.symbolic as sf
from symforce.test_util import TestCase
from symforce.test_util import epsilon_handling
from symforce.test_util import expected_failure_on_sympy
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoMatrixTest(LieGroupOpsTestMixin, TestCase):
    """
    Test the Matrix geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls) -> sf.Matrix:
        return sf.Matrix([-0.2, 5.3, 1.2])

    def test_construction(self) -> None:
        """
        Tests:
            Matrix.__new__
        """
        # Numbers here match Matrix.__new__

        # 1) Matrix32()  # Zero constructed Matrix32
        self.assertEqual(sf.M32(), sf.M([[0, 0], [0, 0], [0, 0]]))

        # 2) Matrix(sf.sympy.Matrix([[1, 2], [3, 4]]))  # Matrix22 with [1, 2, 3, 4] data
        self.assertIsInstance(sf.M(sf.sympy.Matrix([[1, 2], [3, 4]])), sf.M22)
        self.assertEqual(sf.M(sf.sympy.Matrix([[1, 2], [3, 4]])), sf.M([[1, 2], [3, 4]]))
        self.assertRaises(AssertionError, lambda: sf.V3(sf.V2()))

        # 3A) Matrix([[1, 2], [3, 4]])  # Matrix22 with [1, 2, 3, 4] data
        self.assertIsInstance(sf.M([[1, 2], [3, 4]]), sf.M22)
        self.assertEqual(sf.M([[1, 2], [3, 4]]), sf.M22([1, 2, 3, 4]))
        self.assertRaises(AssertionError, lambda: sf.M([[1, 2], [3, 4, 5]]))
        self.assertRaises(AssertionError, lambda: sf.M([[sf.M22(), sf.M23()]]))
        self.assertRaises(AssertionError, lambda: sf.M11([[1, 2]]))

        # 3B) Matrix22([1, 2, 3, 4])  # Matrix22 with [1, 2, 3, 4] data (must matched fixed shape)
        self.assertIsInstance(sf.M22([1, 2, 3, 4]), sf.M22)
        self.assertEqual(sf.M22([1, 2, 3, 4]).to_flat_list(), [1, 2, 3, 4])
        self.assertRaises(AssertionError, lambda: sf.M22([1, 2, 3]))
        self.assertRaises(AssertionError, lambda: sf.M22([1, 2, 3, 4, 5]))

        # 3C) Matrix([1, 2, 3, 4])  # Matrix41 with [1, 2, 3, 4] data - column vector assumed
        self.assertEqual(sf.M([1, 2, 3, 4]), sf.M([[1], [2], [3], [4]]))
        self.assertEqual(sf.M41([1, 2, 3, 4]), sf.M([[1], [2], [3], [4]]))
        self.assertRaises(AssertionError, lambda: sf.M31([1, 2, 3, 4]))

        # 4) Matrix(4, 3)  # Zero constructed Matrix43
        self.assertEqual(sf.M(4, 3), sf.M43.zero())
        self.assertEqual(sf.M(2, 1), sf.M21.zero())
        self.assertEqual(sf.M(1, 2), sf.M12.zero())

        # 5) Matrix(2, 2, [1, 2, 3, 4])  # Matrix22 with [1, 2, 3, 4] data (first two are shape)
        self.assertEqual(sf.M(2, 2, [1, 2, 3, 4]), sf.M([[1, 2], [3, 4]]))

        # 6) Matrix(2, 2, lambda row, col: row + col)  # Matrix22 with [0, 1, 1, 2] data
        self.assertEqual(sf.M(2, 2, lambda row, col: row + col), sf.M22([0, 1, 1, 2]))

        # 7) Matrix22(1, 2, 3, 4)  # Matrix22 with [1, 2, 3, 4] data (must match fixed length)
        self.assertEqual(sf.M22(1, 2, 3, 4), sf.M22([1, 2, 3, 4]))
        self.assertEqual(sf.V4(1, 2, 3, 4), sf.V4([1, 2, 3, 4]))
        self.assertEqual(sf.M21(4, 3), sf.M([[4], [3]]))
        self.assertEqual(sf.M12(4, 3), sf.M([[4, 3]]))
        self.assertEqual(sf.M(4, 3), sf.M.zeros(4, 3))
        self.assertRaises(AssertionError, lambda: sf.M22(1, 2, 3))
        self.assertRaises(AssertionError, lambda: sf.M22(1, 2, 3, 4, 5))

        # Test large size (not statically defined)
        self.assertEqual(type(sf.M(12, 4)).__name__, "Matrix12_4")
        self.assertEqual(type(sf.M(1, 41)).__name__, "Matrix1_41")
        self.assertEqual(type(sf.M(142, 432)).__name__, "Matrix142_432")

        # TODO(hayk): For some reason the symengine constructor slows down to several seconds
        # when the matrix size grows this big. Investigate.
        # self.assertEqual(type(sf.M(1420, 4332)).__name__, 'Matrix1420_4332')

    def test_matrix_initialization(self) -> None:
        """
        Tests:
            Matrix.zero
            Matrix.zeros
            Matrix.one
            Matrix.ones
            Matrix.diag
            Matrix.eye
            Matrix.column_stack
        """
        element = GeoMatrixTest.element()

        dims = element.SHAPE
        self.assertEqual(element.zero(), sf.Matrix.zeros(dims[0], dims[1]))
        self.assertEqual(element.one(), sf.Matrix.ones(dims[0], dims[1]))
        self.assertNotEqual(element.one(), element.zero())

        eye_matrix = sf.Matrix.diag([1, 1, 1])
        self.assertEqual(eye_matrix, sf.Matrix.eye(3))

        nonsquare_diag_matrix = sf.Matrix([[1, 0, 0], [0, 1, 0]])
        self.assertEqual(nonsquare_diag_matrix, sf.Matrix.eye(2, 3))

        vec1 = sf.V3([1, 2, 3])
        vec2 = sf.V3([4, 5, 6])
        vec3 = sf.V3([7, 8, 9])
        mat = sf.Matrix([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        self.assertEqual(mat, sf.Matrix.column_stack(vec1, vec2, vec3))

        # Matrices uniquely allow vector storage
        self.assertEqual(sf.M21.from_storage(sf.M12(3, 4)), sf.M21(3, 4))
        self.assertEqual(sf.M21.from_storage(sf.M21(3, 4)), sf.M21(3, 4))
        self.assertEqual(sf.M12.from_storage(sf.M12(3, 4)), sf.M12(3, 4))
        self.assertEqual(sf.M12.from_storage(sf.M21(3, 4)), sf.M12(3, 4))

    def test_matrix_operations(self) -> None:
        """
        Tests:
            Matrix.inv()
            Matrix.__add__ (scalar_like)
            Matrix.__div__ (scalar_like)
            Matrix.det()
        """

        test_matrix = sf.Matrix([[2, 0], [0, 4]])
        inv_matrix = test_matrix.inv()
        self.assertEqual(sf.Matrix.eye(2), test_matrix * inv_matrix)

        self.assertEqual(sf.Matrix.ones(2, 2), sf.Matrix.zeros(2, 2) + 1)

        diag_matrix = 2 * sf.Matrix.eye(2)
        self.assertEqual(sf.Matrix.eye(2), diag_matrix / 2)
        self.assertEqual(sf.Matrix.eye(2), sf.Matrix(test_matrix / test_matrix))

        self.assertEqual(diag_matrix.det(), 4)

    def test_symbolic_operations(self) -> None:
        """
        Tests:
            Matrix.symbolic
            Matrix.subs
            Matrix.simplify
            Matrix.limit
            Matrix.evalf
            Matrix.to_numpy
        """

        sym_vector = sf.V3.symbolic("vector")
        self.assertEqual(sym_vector.SHAPE, (3, 1))
        sym_matrix = sf.M33.symbolic("matrix")
        self.assertEqual(sym_matrix.SHAPE, (3, 3))

        # Check substitution with the whole matrix as a key
        num_vector = sf.V3(1, 2, -3.1)
        self.assertEqual(num_vector, sym_vector.subs(sym_vector, num_vector))
        norm = sf.S(sym_vector.norm())
        self.assertStorageNear(num_vector.norm(), norm.subs(sym_vector, num_vector), places=9)
        self.assertStorageNear(num_vector.norm(), norm.subs({sym_vector: num_vector}), places=9)
        self.assertStorageNear(num_vector.norm(), norm.subs([(sym_vector, num_vector)]), places=9)
        self.assertStorageNear(
            num_vector * num_vector.T,
            (sym_vector * sym_vector.T).subs(sym_vector, num_vector),
            places=9,
        )

        x = sf.Symbol("x")
        unsimple_matrix = sf.Matrix([x**2 - x, 0])
        simple_matrix = sf.Matrix([x * (x - 1), 0])
        self.assertEqual(unsimple_matrix.simplify(), simple_matrix)

        y = sf.Symbol("y")
        mat = sf.Matrix([x * y / (x + y), (x + y) / (x * y + 1), sf.sin(y) / y])
        mat_lim_y_to_0 = sf.Matrix([0, x, 1])
        self.assertEqual(mat.limit(y, 0), mat_lim_y_to_0)

        pi_mat = sf.pi * sf.M22.eye()
        pi_mat_num = sf.Matrix([[3.14159265, 0], [0, 3.14159265]])
        self.assertStorageNear(pi_mat, pi_mat_num)

        numpy_mat = np.array([[2.0, 1.0], [4.0, 3.0]])
        geo_mat = sf.Matrix([[2.0, 1.0], [4.0, 3.0]])
        self.assertStorageNear(numpy_mat, geo_mat.to_numpy())

        # Make sure we assert when calling a method that expects fixed size on sf.M
        self.assertRaises(AssertionError, lambda: sf.M.symbolic("C"))
        self.assertRaises(AssertionError, lambda: sf.M.from_storage([1, 2, 3]))

    def test_constructor_helpers(self) -> None:
        """
        Tests:
            VectorX, ZX, IX constructor helpers
        """
        vector_constructors = [sf.V1, sf.V2, sf.V3, sf.V4, sf.V5, sf.V6]
        for i, vec in enumerate(vector_constructors):
            self.assertEqual(vec(), sf.Matrix.zeros(i + 1, 1))
            rand_vec = np.random.rand(i + 1)
            self.assertEqual(vec(rand_vec), sf.Matrix(rand_vec))
            self.assertEqual(vec(vec(rand_vec)), sf.Matrix(rand_vec))
            self.assertEqual(vec(*rand_vec), sf.Matrix(rand_vec))

            rand_vec_long = np.random.rand(i + 2)
            self.assertRaises(AssertionError, vec, rand_vec_long)
            self.assertRaises(AssertionError, vec, *rand_vec_long)

        eye_matrix_constructors = [sf.I1, sf.I2, sf.I3, sf.I4, sf.I5, sf.I6]
        for i, mat in enumerate(eye_matrix_constructors):
            self.assertEqual(mat(), sf.Matrix.eye(i + 1))

    def test_row_col_join(self) -> None:
        """
        Tests:
            row_join
            col_join
        """
        self.assertEqual(sf.M33().SHAPE, sf.M32().row_join(sf.M31()).SHAPE)
        self.assertEqual(sf.M33().SHAPE, sf.M23().col_join(sf.M13()).SHAPE)

    def test_jacobian(self) -> None:
        """
        Tests:
            Matrix.jacobian
        """
        vec = sf.V3.symbolic("vec")
        pose = sf.Pose3.symbolic("pose")
        new_vec = pose * vec

        vec_D_pose = vec.jacobian(pose)
        self.assertEqual(vec_D_pose, sf.Matrix(3, pose.tangent_dim()).zero())

        new_vec_D_vec = new_vec.jacobian(vec)
        new_vec_D_pose = new_vec.jacobian(pose)
        self.assertEqual(new_vec_D_vec.shape, (3, 3))
        self.assertEqual(new_vec_D_pose.shape, (3, pose.tangent_dim()))

        vec_D_pose_storage = vec.jacobian(pose, tangent_space=False)
        self.assertEqual(vec_D_pose_storage, sf.Matrix(3, pose.storage_dim()).zero())

        x = sf.M21.symbolic("x")
        self.assertEqual(sf.M22.eye(), x.jacobian(x))

        mat = sf.M22.symbolic("a")
        vec = mat[:, 0]
        self.assertRaises(AssertionError, lambda: mat.jacobian(vec))

    def test_block_matrix(self) -> None:
        """
        Tests:
            Matrix.block_matrix
        """
        M22 = sf.M22([1, 1, 1, 1])
        M23 = sf.M23([2, 2, 2, 2, 2, 2])
        M11 = sf.M11([3])
        M14 = sf.M14([4, 4, 4, 4])
        self.assertEqual(
            sf.M.block_matrix([[M22, M23], [M11, M14]]),
            sf.M([[1, 1, 2, 2, 2], [1, 1, 2, 2, 2], [3, 4, 4, 4, 4]]),
        )
        M21 = sf.M21([5, 5])
        M13 = sf.M13([6, 6, 6])
        self.assertEqual(
            sf.M.block_matrix([[M22, M21], [M13]]), sf.M([[1, 1, 5], [1, 1, 5], [6, 6, 6]])
        )
        self.assertRaises(AssertionError, lambda: sf.M.block_matrix([[M22, M23], [M11, sf.M15()]]))
        self.assertRaises(AssertionError, lambda: sf.M.block_matrix([[M22, sf.M33()], [M11, M14]]))

    def test_transpose(self) -> None:
        """
        Tests:
            Matrix.T
        """
        m22 = sf.M22([1, 2, 3, 4])
        self.assertEqual(m22.T, sf.M22([1, 3, 2, 4]))

        m21 = sf.M21([1, 2])
        self.assertEqual(m21.T, sf.M12([1, 2]))

        m12 = sf.M12([1, 2])
        self.assertEqual(m12.T, sf.M21([1, 2]))

        self.assertEqual(sf.M12() * sf.M21(), sf.M11())
        self.assertEqual(sf.M21() * sf.M12(), sf.M22())

    def test_lower_triangle(self) -> None:
        """
        Tests:
            Matrix.lower_triangle
        """
        m11 = sf.M11(1)
        self.assertEqual(m11, m11.lower_triangle())

        m22 = sf.M22([[1, 2], [3, 4]])
        self.assertEqual(sf.M22([[1, 0], [3, 4]]), m22.lower_triangle())

        m33 = sf.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(sf.M33([[1, 0, 0], [4, 5, 0], [7, 8, 9]]), m33.lower_triangle())

        self.assertRaises(ValueError, lambda: sf.M45().lower_triangle())
        self.assertRaises(ValueError, lambda: sf.M54().lower_triangle())

    def test_symmetric_copy(self) -> None:
        """
        Tests:
            Matrix.symmetric_copy
        """

        m22 = sf.M22([[1, 2], [3, 4]])
        self.assertEqual(m22.symmetric_copy(sf.Matrix.Triangle.LOWER), sf.M22([[1, 3], [3, 4]]))
        self.assertEqual(m22.symmetric_copy(sf.Matrix.Triangle.UPPER), sf.M22([[1, 2], [2, 4]]))

        with self.assertRaises(TypeError):
            sf.M31().symmetric_copy(sf.Matrix.Triangle.LOWER)

        m22_copy = m22.symmetric_copy(sf.Matrix.Triangle.LOWER)
        m22_copy[0, 1] = 5
        self.assertNotEqual(m22, m22_copy)

    def test_row_col(self) -> None:
        """
        Tests:
            Matrix.row
            Matrix.col
        """
        m22 = sf.M22([1, 2, 3, 4])
        self.assertEqual(m22.row(0), sf.M12([1, 2]))
        self.assertEqual(m22.col(0), sf.M21([1, 3]))

    def test_clamp_norm(self) -> None:
        """
        Tests:
            Matrix.clamp_norm
        """
        v = sf.V3(1, 2, 3)

        with self.subTest("max_norm = 2"):
            v_clamped = v.clamp_norm(2)
            self.assertStorageNear(v_clamped, v.normalized() * 2)

        with self.subTest("max_norm = 10"):
            v_clamped = v.clamp_norm(10)
            self.assertStorageNear(v_clamped, v)

        if symforce.get_symbolic_api() == "sympy":
            with self.subTest("epsilon handling"):

                def scalar_clamp_norm(x: sf.Scalar, epsilon: sf.Scalar) -> sf.Scalar:
                    return sf.V3(0, 0, x).clamp_norm(2, epsilon)[2, 0]

                # TODO: Also test the derivative
                # SymPy is currently incapable of evaluating the derivative here
                self.assertTrue(
                    epsilon_handling.is_value_with_epsilon_correct(
                        scalar_clamp_norm, expected_value=0
                    )
                )

    def test_multiply_elementwise(self) -> None:
        """
        Tests:
            Matrix.multiply_elementwise
        """
        a = sf.M34()
        a[0, 1] = 1
        a[1, 0] = 2
        a[2, 3] = 3

        b = sf.M34()
        b[1, 0] = 4
        b[2, 3] = 5

        expected_result = sf.M34()
        expected_result[1, 0] = 2 * 4
        expected_result[2, 3] = 3 * 5

        self.assertEqual(a.multiply_elementwise(b), expected_result)

        with self.assertRaises(AssertionError):
            # This should fail mypy, since it's actually wrong
            a.multiply_elementwise(sf.M43())  # type: ignore

    def test_dot_product(self) -> None:
        """
        Tests:
            Matrix.dot
        """

        a = sf.V2()
        a[0] = 1
        a[1] = 2

        b = sf.V2()
        b[0] = 3
        b[1] = 4

        expected_result = 3 + 2 * 4
        self.assertEqual(a.dot(b), expected_result)
        # NOTE(harrison): update this if we start supporting complex
        self.assertEqual(b.dot(a), expected_result)
        self.assertEqual(a.T.dot(b.T), expected_result)

        with self.assertRaises(TypeError):
            c = sf.M33()
            d = sf.M33()
            c.dot(d)
        with self.assertRaises(TypeError):
            self.assertEqual(a.T.dot(b), expected_result)
        with self.assertRaises(TypeError):
            self.assertEqual(a.dot(b.T), expected_result)

    @expected_failure_on_sympy
    def test_setitem(self) -> None:
        """
        Tests:
            Matrix.__setitem__
        """
        with self.subTest(msg="Can assign a 1d array to a 1d slice"):
            m = sf.M12()
            m[0, :] = [1, 2]
            self.assertEqual(m, sf.M12([1, 2]))

        with self.subTest(msg="Can assign List[np.float64] to tupled index entry"):
            m = sf.M12()
            m[0, :] = [np.float64(1), np.float64(2)]
            self.assertEqual(m, sf.M12([1.0, 2.0]))

        with self.subTest(msg="Assigning a too small list to a tupled index entry"):
            m = sf.M12()
            with self.assertRaises(IndexError):
                m[0, :] = [np.float64(1)]

        with self.subTest(msg="Assigning a too small 2d array to tupled index entry"):
            # NOTE(brad): These tests are run to ensure that the fix to keep an IndexError
            # from being raised for np.float64s doesn't suppress IndexErrors where they
            # are warranted.
            m22 = sf.M22()
            with self.assertRaises(IndexError):
                m22[:, :] = [[np.float64(1), np.float64(2)]]

            with self.assertRaises(IndexError):
                m22[:, :] = [[np.float64(1)], [np.float64(2)]]

    def test_getitem(self) -> None:
        """
        Test:
            Matrix.__getitem__
        """
        # NOTE(nathan): This differs from how a sympy matrix behaves, which would return a list
        # object when sliced
        m31 = sf.M31([1, 2, 3])
        self.assertEqual(m31[0:2], sf.M21([1, 2]))
        m13 = sf.M13([1, 2, 3])
        self.assertEqual(m13[0:2], sf.M12([1, 2]))

    def test_vector_methods(self) -> None:
        """
        Tests:
            V2.x,y
            V3.x,y,z
            V2.unit_x,y
            V3.unit_x,y,z
        """
        self.assertEqual(sf.V2.unit_x().x, 1)
        self.assertEqual(sf.V2.unit_x().y, 0)
        with self.assertRaises(AttributeError):
            _ = sf.V2.unit_x().z  # type: ignore[attr-defined]  # pylint: disable=no-member

        self.assertEqual(sf.V2.unit_y().x, 0)
        self.assertEqual(sf.V2.unit_y().y, 1)

        with self.assertRaises(AttributeError):
            sf.V2.unit_z()  # type: ignore[attr-defined]  # pylint: disable=no-member

        self.assertEqual(sf.V3.unit_x().x, 1)
        self.assertEqual(sf.V3.unit_x().y, 0)
        self.assertEqual(sf.V3.unit_x().z, 0)

        self.assertEqual(sf.V3.unit_y().x, 0)
        self.assertEqual(sf.V3.unit_y().y, 1)
        self.assertEqual(sf.V3.unit_y().z, 0)

        self.assertEqual(sf.V3.unit_z().x, 0)
        self.assertEqual(sf.V3.unit_z().y, 0)
        self.assertEqual(sf.V3.unit_z().z, 1)


if __name__ == "__main__":
    TestCase.main()
