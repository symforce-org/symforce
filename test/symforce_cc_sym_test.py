# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import dataclasses
import math
import pickle

import numpy as np
from scipy import sparse

from lcmtypes.sym._index_entry_t import index_entry_t
from lcmtypes.sym._index_t import index_t
from lcmtypes.sym._key_t import key_t
from lcmtypes.sym._levenberg_marquardt_solver_failure_reason_t import (
    levenberg_marquardt_solver_failure_reason_t,
)
from lcmtypes.sym._linearized_dense_factor_t import linearized_dense_factor_t
from lcmtypes.sym._optimization_iteration_t import optimization_iteration_t
from lcmtypes.sym._optimization_stats_t import optimization_stats_t
from lcmtypes.sym._optimization_status_t import optimization_status_t
from lcmtypes.sym._optimizer_params_t import optimizer_params_t
from lcmtypes.sym._values_t import values_t

import sym
from symforce import cc_sym
from symforce import typing as T
from symforce.opt import optimizer
from symforce.test_util import TestCase


class SymforceCCSymTest(TestCase):
    """
    Test cc_sym.
    """

    def test_key(self) -> None:
        """
        Tests:
            cc_sym.Key
        """
        with self.subTest(msg="static member fields were wrapped"):
            self.assertIsInstance(cc_sym.Key.INVALID_LETTER, str)
            self.assertIsInstance(cc_sym.Key.INVALID_SUB, int)
            self.assertIsInstance(cc_sym.Key.INVALID_SUPER, int)

        with self.subTest(msg="Two keys with the same fields are equal"):
            self.assertEqual(cc_sym.Key("a"), cc_sym.Key("a"))
            self.assertEqual(cc_sym.Key("a", 1), cc_sym.Key("a", 1))
            self.assertEqual(cc_sym.Key("a", 1, 2), cc_sym.Key("a", 1, 2))

        with self.subTest(msg="Two keys with different fields are not equal"):
            self.assertNotEqual(cc_sym.Key("a"), cc_sym.Key("b"))

        with self.subTest(msg="A key is not equal to instances of other types"):
            self.assertNotEqual(cc_sym.Key("a"), 1)
            self.assertNotEqual("a", cc_sym.Key("a"))

        with self.subTest(msg="A key can be specified with keyword arguments"):
            self.assertEqual(cc_sym.Key("a"), cc_sym.Key(letter="a"))
            self.assertEqual(cc_sym.Key("a", 1), cc_sym.Key(letter="a", sub=1))
            self.assertEqual(cc_sym.Key("a", 1, 2), cc_sym.Key(letter="a", sub=1, super=2))

        with self.subTest(msg="Accessors correctly return the fields"):
            key = cc_sym.Key("a", 1, 2)
            self.assertEqual(key.letter, "a")
            self.assertEqual(key.sub, 1)
            self.assertEqual(key.super, 2)

        with self.subTest(msg="Method with_letter works as intended"):
            key = cc_sym.Key(letter="a", sub=1, super=2)

            new_letter = "b"
            new_key = key.with_letter(letter=new_letter)
            self.assertEqual(new_key.letter, new_letter)
            self.assertEqual(new_key.sub, key.sub)
            self.assertEqual(new_key.super, key.super)

        with self.subTest(msg="Method with_sub works as intended"):
            key = cc_sym.Key(letter="a", sub=1, super=2)

            new_sub = 3
            new_key = key.with_sub(sub=new_sub)
            self.assertEqual(new_key.letter, key.letter)
            self.assertEqual(new_key.sub, new_sub)
            self.assertEqual(new_key.super, key.super)

        with self.subTest(msg="Method with_super works as intended"):
            key = cc_sym.Key(letter="a", sub=1, super=2)

            new_super = 4
            new_key = key.with_super(super=new_super)
            self.assertEqual(new_key.letter, key.letter)
            self.assertEqual(new_key.sub, key.sub)
            self.assertEqual(new_key.super, new_super)

        letter_sub_super_samples: T.List[
            T.Union[T.Tuple[str], T.Tuple[str, int], T.Tuple[str, int, int]]
        ] = []
        for letter in ["a", "b"]:
            letter_sub_super_samples.append((letter,))
            for sub in [1, 2]:
                letter_sub_super_samples.append((letter, sub))
                for sup in [3, 4]:
                    letter_sub_super_samples.append((letter, sub, sup))

        with self.subTest(msg="inequality operators match that of tuples"):
            for tuple1 in letter_sub_super_samples:
                for tuple2 in letter_sub_super_samples:
                    self.assertEqual(
                        cc_sym.Key(*tuple1).lexical_less_than(cc_sym.Key(*tuple2)), tuple1 < tuple2
                    )

        with self.subTest(msg="cc_sym.Key.__hash__ is defined"):
            hash(cc_sym.Key("a"))

        with self.subTest(msg="cc_sym.Key.get_lcm_type returns a key_t"):
            self.assertIsInstance(cc_sym.Key("a").get_lcm_type(), key_t)

        with self.subTest(msg="cc_sym.Key is pickleable"):
            for key in [
                cc_sym.Key("a"),
                cc_sym.Key("a", 1),
                cc_sym.Key("a", 1, 2),
                cc_sym.Key("a", cc_sym.Key.INVALID_SUB, 2),
            ]:
                key_dumps = pickle.dumps(key)
                self.assertEqual(key, pickle.loads(key_dumps))

    def test_values(self) -> None:
        """
        Tests:
            cc_sym.Values
            cc_sym.Key
            Implicitly tests conversions of sym types:
                sym.Rot2
                sym.Rot3
                sym.Pose2
                sym.Pose3
                sym.Unit3
                sym.ATANCameraCal
                sym.DoubleSphereCameraCal
                sym.EquirectangularCameraCal
                sym.LinearCameraCal
                sym.PolynomialCameraCal
                sym.SphericalCameraCal
        """

        supported_types = [
            T.Scalar,
            sym.Rot2,
            sym.Rot3,
            sym.Pose2,
            sym.Pose3,
            sym.Unit3,
            sym.ATANCameraCal,
            sym.DoubleSphereCameraCal,
            sym.EquirectangularCameraCal,
            sym.LinearCameraCal,
            sym.PolynomialCameraCal,
            sym.SphericalCameraCal,
        ]

        def instantiate_type(tp: T.Type[T.Any]) -> T.Any:
            """
            Helper to instantiate tp. Useful for getting an instance of tp to test storage
            of that type in cc_sym.Values
            """
            try:
                return tp()
            except TypeError:
                # The camera cals do not have a default constructor, so we construct it
                # from storage instead.
                return tp.from_storage([0] * tp.storage_dim())

        for tp in supported_types:
            with self.subTest(
                msg=f"Can set and retrieve {tp.__name__}"  # pylint: disable=no-member
            ):
                values = cc_sym.Values()
                val: T.Any = instantiate_type(tp)
                values.set(cc_sym.Key("v"), val)
                self.assertEqual(values.at(cc_sym.Key("v")), val)

        with self.subTest(msg="Can set and at 9x9 matrices and smaller"):
            values = cc_sym.Values()
            for rows in range(1, 10):
                for cols in range(1, 10):
                    matrix = np.array([[0] * cols] * rows)
                    values.set(cc_sym.Key("l", rows, cols), matrix)
                    values.at(cc_sym.Key("l", rows, cols))

                    values.set(cc_sym.Key("a", rows, cols), np.array(matrix))
                    values.at(cc_sym.Key("a", rows, cols))

        with self.subTest(msg="at raises RuntimeError if no entry exists"):
            with self.assertRaises(RuntimeError):
                cc_sym.Values().at(cc_sym.Key("a"))

        with self.subTest(msg="set returns true no value existed yet for the key"):
            values = cc_sym.Values()
            self.assertTrue(values.set(cc_sym.Key("a"), 1))
            self.assertFalse(values.set(cc_sym.Key("a"), 2))

        with self.subTest(msg="has returns whether or not key is present in Values"):
            values = cc_sym.Values()
            key = cc_sym.Key("a")
            self.assertFalse(values.has(key))
            values.set(key, 1)
            self.assertTrue(values.has(key))

        with self.subTest(msg="test that Remove returns whether or not key to be removed existed"):
            values = cc_sym.Values()
            key = cc_sym.Key("a")
            self.assertFalse(values.remove(key))
            values.set(key, 3)
            self.assertTrue(values.remove(key))

        with self.subTest(msg="Test that remove is consistent with has"):
            values = cc_sym.Values()
            key = cc_sym.Key("a")
            values.set(key, 1)
            values.remove(key=key)
            self.assertFalse(values.has(key))

        with self.subTest(msg="num_entries returns the correct number of entries"):
            values = cc_sym.Values()
            self.assertEqual(values.num_entries(), 0)
            values.set(cc_sym.Key("a"), 1.2)
            self.assertEqual(values.num_entries(), 1)
            values.remove(cc_sym.Key("a"))
            self.assertEqual(values.num_entries(), 0)

        with self.subTest(msg="Values.empty returns true if empty and false otherwise"):
            values = cc_sym.Values()
            self.assertTrue(values.empty())
            values.set(cc_sym.Key("a"), 1)
            self.assertFalse(values.empty())

        with self.subTest("Values.keys works correctly"):
            values = cc_sym.Values()
            a = cc_sym.Key("a")
            a_1 = cc_sym.Key("a", 1)
            b = cc_sym.Key("b")
            values.set(a_1, 1)
            values.set(b, 2)
            values.set(a, 3)

            self.assertEqual([a_1, b, a], values.keys())
            self.assertEqual([a_1, b, a], values.keys(sort_by_offset=True))
            keys_false = values.keys(sort_by_offset=False)
            self.assertEqual({a, a_1, b}, set(keys_false))
            self.assertEqual(3, len(keys_false))

        with self.subTest("Values.items returns a dict[Key, index_entry_t]"):
            values = cc_sym.Values()
            a = cc_sym.Key("a")
            values.set(a, 1)
            items = values.items()
            self.assertIsInstance(items, dict)
            self.assertIn(a, items)
            self.assertIsInstance(items[a], index_entry_t)

        with self.subTest("Values.items and Values.at [index_entry_t version] work together"):
            values = cc_sym.Values()
            a = cc_sym.Key("a")
            values.set(a, 1)
            items = values.items()
            self.assertEqual(values.at(entry=items[a]), 1)

        with self.subTest("Values.data returns the correct value"):
            values = cc_sym.Values()
            values.set(cc_sym.Key("a"), 1)
            values.set(cc_sym.Key("b"), 2)
            self.assertEqual(values.data(), [1, 2])

        with self.subTest(msg="Values.create_index returns an index_t"):
            values = cc_sym.Values()
            keys = [cc_sym.Key("a", i) for i in range(10)]
            for key in keys:
                values.set(key, key.sub)
            self.assertIsInstance(values.create_index(keys=keys), index_t)

        with self.subTest(msg="Values.update_or_set works as expected"):
            key_a = cc_sym.Key("a")
            key_b = cc_sym.Key("b")
            key_c = cc_sym.Key("c")

            values_1 = cc_sym.Values()
            values_1.set(key_a, 1)
            values_1.set(key_b, 2)

            values_2 = cc_sym.Values()
            values_2.set(key_b, 3)
            values_2.set(key_c, 4)

            values_1.update_or_set(index=values_2.create_index([key_b, key_c]), other=values_2)

            self.assertEqual(values_1.at(key_a), 1)
            self.assertEqual(values_1.at(key_b), 3)
            self.assertEqual(values_1.at(key_c), 4)

        with self.subTest(msg="Values.remove_all leaves a values as empty"):
            values = cc_sym.Values()
            for i in range(4):
                values.set(cc_sym.Key("a", i), i)
            values.remove_all()
            self.assertTrue(values.empty())

        with self.subTest(msg="Test that Values.cleanup is callable and returns correct output"):
            values = cc_sym.Values()
            values.set(cc_sym.Key("a"), 1)
            values.set(cc_sym.Key("b"), 2)
            values.remove(cc_sym.Key("a"))
            self.assertEqual(values.cleanup(), 1)

        for tp in supported_types:
            with self.subTest(
                msg=f"Can call set as a function of index_entry_t and {tp.__name__}"  # pylint: disable=no-member
            ):
                values = cc_sym.Values()
                a = cc_sym.Key("a")
                values.set(a, instantiate_type(tp))
                values.set(values.items()[a], instantiate_type(tp))

        with self.subTest(msg="Test Values.update (since index overlaod) works as expected"):
            key_a = cc_sym.Key("a")
            key_b = cc_sym.Key("b")
            key_c = cc_sym.Key("c")

            values_1 = cc_sym.Values()
            values_1.set(key_a, 1)
            values_1.set(key_b, 2)
            values_1.set(key_c, 3)

            values_2 = cc_sym.Values()
            values_2.set(key_a, 4)
            values_2.set(key_b, 5)
            values_2.set(key_c, 6)

            values_1.update(index=values_1.create_index([key_b, key_c]), other=values_2)

            self.assertEqual(values_1.at(key_a), 1)
            self.assertEqual(values_1.at(key_b), 5)
            self.assertEqual(values_1.at(key_c), 6)

        with self.subTest(msg="Test Values.update (two index overlaod) works as expected"):
            key_a = cc_sym.Key("a")
            key_b = cc_sym.Key("b")
            key_c = cc_sym.Key("c")

            values_1 = cc_sym.Values()
            values_1.set(key_a, 1)
            values_1.set(key_b, 2)
            values_1.set(key_c, 3)

            values_2 = cc_sym.Values()
            values_2.set(key_b, 4)
            values_2.set(key_c, 5)

            values_1.update(
                index_this=values_1.create_index([key_b, key_c]),
                index_other=values_2.create_index([key_b, key_c]),
                other=values_2,
            )

            self.assertEqual(values_1.at(key_a), 1)
            self.assertEqual(values_1.at(key_b), 4)
            self.assertEqual(values_1.at(key_c), 5)

        with self.subTest(msg="Test that Values.retract works roughly"):
            a = cc_sym.Key("a")
            values_1 = cc_sym.Values()
            values_1.set(a, 0)
            values_2 = cc_sym.Values()
            values_2.set(a, 0)

            values_2.retract(values_2.create_index([a]), [1], 1e-8)

            self.assertNotEqual(values_1, values_2)

        with self.subTest(msg="Test that Values.local_coordinates works roughly"):
            a = cc_sym.Key("a")
            values_1 = cc_sym.Values()
            values_1.set(a, 0)
            values_2 = cc_sym.Values()
            values_2.set(a, 10)
            self.assertEqual(
                values_2.local_coordinates(values_1, values_1.create_index([a]), 0), 10
            )

        with self.subTest(msg="Test that Values.get_lcm_type returns a values_t"):
            v = cc_sym.Values()
            v.set(cc_sym.Key("a", 1, 2), 10)
            self.assertIsInstance(v.get_lcm_type(), values_t)

        with self.subTest(msg="Test the initializer from values_t"):
            v = cc_sym.Values()
            a = cc_sym.Key("a")
            v.set(cc_sym.Key("a"), 1)
            v_copy = cc_sym.Values(v.get_lcm_type())
            self.assertTrue(v_copy.has(a))
            self.assertEqual(v_copy.at(a), v.at(a))

        with self.subTest(msg="Can pickle Values"):
            v = cc_sym.Values()
            keys = []
            for i, tp in enumerate(supported_types):
                v.set(cc_sym.Key("x", i), instantiate_type(tp))
                keys.append(cc_sym.Key("x", i))

            pickled_v = pickle.loads(pickle.dumps(v))

            for key in keys:
                self.assertEqual(v.at(key), pickled_v.at(key))

    @staticmethod
    def pi_residual(
        x: T.Scalar,
    ) -> T.Tuple[T.List[T.Scalar], T.List[T.Scalar], T.List[T.Scalar], T.List[T.Scalar]]:
        """
        Numerical residual function r(x) = cos(x / 2) with linearization

        Args:
            x: Scalar

        Outputs:
            res: Matrix11
            jacobian: (1x1) jacobian of res wrt arg x (1)
            hessian: (1x1) Gauss-Newton hessian for arg x (1)
            rhs: (1x1) Gauss-Newton rhs for arg x (1)
        """
        x_2 = x / 2.0
        cos = math.cos(x_2)
        sin = math.sin(x_2)
        sin_2 = (1.0 / 2.0) * sin

        # Output terms
        res = [cos]
        jacobian = [-sin_2]
        hessian = [(1.0 / 4.0) * sin ** 2]
        rhs = [-cos * sin_2]
        return res, jacobian, hessian, rhs

    @staticmethod
    def sparse_pi_residual(
        x: T.Scalar,
    ) -> T.Tuple[T.List[T.Scalar], sparse.csc_matrix, sparse.csc_matrix, T.List[T.Scalar]]:
        """
        Same as pi_residual, but the jacobian and hessian are scipy.sparse.csc_matrix's
        """
        res, jacobian, hessian, rhs = SymforceCCSymTest.pi_residual(x)
        return res, sparse.csc_matrix(jacobian), sparse.csc_matrix(hessian), rhs

    dense_pi_hessian = lambda values, index_entries: SymforceCCSymTest.pi_residual(
        values.at(index_entries[0])
    )

    dense_pi_jacobian = lambda values, index_entries: SymforceCCSymTest.pi_residual(
        values.at(index_entries[0])
    )[0:2]

    sparse_pi_hessian = lambda values, index_entries: SymforceCCSymTest.sparse_pi_residual(
        values.at(index_entries[0])
    )

    sparse_pi_jacobian = lambda values, index_entries: SymforceCCSymTest.sparse_pi_residual(
        values.at(index_entries[0])
    )[0:2]

    def test_factor(self) -> None:
        """
        Tests:
            cc_sym.Factor
        """

        pi_key = cc_sym.Key("3", 1, 4)
        pi_factor = cc_sym.Factor(
            hessian_func=SymforceCCSymTest.dense_pi_hessian,
            keys=[pi_key],
        )
        pi_jacobian_factor = cc_sym.Factor.jacobian(
            jacobian_func=SymforceCCSymTest.dense_pi_jacobian,
            keys=[pi_key],
        )

        sparse_pi_factor = cc_sym.Factor(
            hessian_func=SymforceCCSymTest.sparse_pi_hessian,
            keys=[pi_key],
            sparse=True,
        )
        sparse_pi_jacobian_factor = cc_sym.Factor.jacobian(
            jacobian_func=SymforceCCSymTest.sparse_pi_jacobian, keys=[pi_key], sparse=True
        )

        with self.subTest(msg="Test that alternate Factor constructors can be called"):
            cc_sym.Factor(
                hessian_func=SymforceCCSymTest.dense_pi_hessian,
                keys_to_func=[pi_key],
                keys_to_optimize=[pi_key],
            )

            cc_sym.Factor.jacobian(
                jacobian_func=SymforceCCSymTest.dense_pi_jacobian,
                keys_to_func=[pi_key],
                keys_to_optimize=[pi_key],
                sparse=True,
            )

            cc_sym.Factor(
                hessian_func=SymforceCCSymTest.sparse_pi_hessian,
                keys_to_func=[pi_key],
                keys_to_optimize=[pi_key],
                sparse=True,
            )

            cc_sym.Factor.jacobian(
                jacobian_func=SymforceCCSymTest.sparse_pi_jacobian,
                keys_to_func=[pi_key],
                keys_to_optimize=[pi_key],
                sparse=True,
            )

        # Test that Factor.is_sparse is wrapped
        self.assertFalse(pi_factor.is_sparse())
        self.assertFalse(pi_jacobian_factor.is_sparse())
        self.assertTrue(sparse_pi_factor.is_sparse())
        self.assertTrue(sparse_pi_jacobian_factor.is_sparse())

        with self.subTest(msg="Test that Factor.linearized_factor/linearize are wrapped"):
            pi_values = cc_sym.Values()
            eval_value = 3.0
            pi_values.set(pi_key, eval_value)
            self.assertIsInstance(pi_factor.linearized_factor(pi_values), linearized_dense_factor_t)

            target_residual, target_jacobian, *_ = SymforceCCSymTest.pi_residual(eval_value)

            for factor in [
                pi_factor,
                pi_jacobian_factor,
                sparse_pi_factor,
                sparse_pi_jacobian_factor,
            ]:
                residual, jacobian = factor.linearize(pi_values)
                self.assertAlmostEqual(target_residual[0], residual[0])
                self.assertAlmostEqual(target_jacobian[0], jacobian[0, 0])
                if factor.is_sparse():
                    self.assertIsInstance(jacobian, sparse.csc_matrix)
                else:
                    self.assertNotIsInstance(jacobian, sparse.csc_matrix)

        with self.subTest(msg="Test error is raised if mismatch in sparsity of factor and matrix"):
            pi_values = cc_sym.Values()
            pi_values.set(pi_key, 3.0)

            sparse_factor_dense_hessian = cc_sym.Factor(
                hessian_func=SymforceCCSymTest.dense_pi_hessian,
                keys=[pi_key],
                sparse=True,
            )
            with self.assertRaises(ValueError):
                sparse_factor_dense_hessian.linearize(pi_values)

            dense_factor_sparse_hessian = cc_sym.Factor(
                hessian_func=SymforceCCSymTest.sparse_pi_hessian,
                keys=[pi_key],
                sparse=False,
            )
            with self.assertRaises(ValueError):
                dense_factor_sparse_hessian.linearize(pi_values)

            sparse_factor_dense_jacobian = cc_sym.Factor.jacobian(
                jacobian_func=SymforceCCSymTest.dense_pi_jacobian,
                keys=[pi_key],
                sparse=True,
            )
            with self.assertRaises(ValueError):
                sparse_factor_dense_jacobian.linearize(pi_values)

            dense_factor_sparse_jacobian = cc_sym.Factor.jacobian(
                jacobian_func=SymforceCCSymTest.sparse_pi_jacobian,
                keys=[pi_key],
                sparse=False,
            )
            with self.assertRaises(ValueError):
                dense_factor_sparse_jacobian.linearize(pi_values)

        with self.subTest(msg="Test that Factor.all_keys and optimized_keys are wrapped"):
            self.assertEqual(pi_factor.all_keys(), [pi_key])
            self.assertEqual(pi_factor.optimized_keys(), [pi_key])

    @staticmethod
    def compare_linearizations(lin1: cc_sym.Linearization, lin2: cc_sym.Linearization) -> bool:
        return (
            (lin1.residual == lin2.residual).all()
            and (lin1.hessian_lower.toarray() == lin2.hessian_lower.toarray()).all()
            and (lin1.jacobian.toarray() == lin2.jacobian.toarray()).all()
            and (lin1.rhs == lin2.rhs).all()
            and lin1.is_initialized() == lin2.is_initialized()
        )

    @staticmethod
    def compare_optimization_stats(
        stats1: cc_sym.OptimizationStats, stats2: cc_sym.OptimizationStats
    ) -> bool:

        TVar = T.TypeVar("TVar")

        # NOTE(brad): Exists to make mypy happy
        def unwrap(option: T.Optional[TVar]) -> TVar:
            assert option is not None
            return option

        if (stats1.best_linearization is None) ^ (stats2.best_linearization is None):
            return False
        return (
            stats1.iterations == stats2.iterations
            and stats1.best_index == stats2.best_index
            and stats1.status == stats2.status
            and stats1.failure_reason == stats2.failure_reason
            and (
                stats1.best_linearization is None
                and stats2.best_linearization is None
                or SymforceCCSymTest.compare_linearizations(
                    unwrap(stats1.best_linearization), unwrap(stats2.best_linearization)
                )
            )
        )

    def test_optimization_stats(self) -> None:
        """
        Tests:
            cc_sym.OptimizationStats
            cc_sym.Linearization
        """

        with self.subTest(msg="Can read and write to iterations field"):
            stats = cc_sym.OptimizationStats()
            self.assertIsInstance(stats.iterations, list)
            stats.iterations = [optimization_iteration_t() for _ in range(5)]

        with self.subTest(msg="Can read and write to best_index and status"):
            stats = cc_sym.OptimizationStats()
            stats.best_index = stats.best_index
            stats.status = stats.status
            stats.failure_reason = stats.failure_reason

        with self.subTest(msg="Can read and write to best_linearization"):
            stats = cc_sym.OptimizationStats()
            stats.best_linearization = None
            self.assertIsNone(stats.best_linearization)
            stats.best_linearization = cc_sym.Linearization()
            self.assertIsInstance(stats.best_linearization, cc_sym.Linearization)

        with self.subTest(msg="get_lcm_type is wrapped"):
            stats = cc_sym.OptimizationStats()
            self.assertIsInstance(stats.get_lcm_type(), optimization_stats_t)

        with self.subTest(msg="Can pickle cc_sym.OptimizationStats"):

            stats = cc_sym.OptimizationStats()
            stats.iterations = [optimization_iteration_t(iteration=i) for i in range(4)]
            stats.best_index = 1
            stats.status = optimization_status_t.SUCCESS
            stats.failure_reason = levenberg_marquardt_solver_failure_reason_t.INVALID.value
            stats.best_linearization = None

            self.assertTrue(
                self.compare_optimization_stats(stats, pickle.loads(pickle.dumps(stats)))
            )

            linearization = cc_sym.Linearization()
            linearization.residual = np.array([1, 2, 3])
            stats.best_linearization = linearization

            self.assertTrue(
                self.compare_optimization_stats(stats, pickle.loads(pickle.dumps(stats)))
            )

    def test_optimizer(self) -> None:
        """
        Tests:
            cc_sym.default_optimizer_params
            cc_sym.Optimizer
            cc_sym.Linearization
        """

        with self.subTest(msg="Test that default_optimizer_params is wrapped"):
            self.assertIsInstance(cc_sym.default_optimizer_params(), optimizer_params_t)

        pi_key = cc_sym.Key("3", 1, 4)
        pi_factor = cc_sym.Factor(
            hessian_func=lambda values, index_entries: SymforceCCSymTest.pi_residual(
                values.at(index_entries[0])
            ),
            keys=[pi_key],
        )

        with self.subTest(msg="Can construct an Optimizer with or without default arguments"):
            cc_sym.Optimizer(params=cc_sym.default_optimizer_params(), factors=[pi_factor])
            cc_sym.Optimizer(
                params=cc_sym.default_optimizer_params(),
                factors=[pi_factor],
                epsilon=1e-6,
                name="OptimizeTest",
                keys=[],
                debug_stats=False,
                check_derivatives=False,
            )

        make_opt = lambda: cc_sym.Optimizer(
            params=cc_sym.default_optimizer_params(),
            factors=[pi_factor],
            debug_stats=False,
            include_jacobians=True,
        )

        with self.subTest(msg="Optimizer.factors has been wrapped"):
            opt = make_opt()

            self.assertEqual(1, len(opt.factors()))
            self.assertEqual(opt.factors()[0].all_keys(), pi_factor.all_keys())

        with self.subTest(msg="Optimizer.optimize has been wrapped"):
            values = cc_sym.Values()
            values.set(pi_key, 3.0)

            opt = make_opt()

            # Testing the wrapping of overload
            # OptimizationStats<Scalar> Optimize(Values<Scalar>* values, int num_iterations = -1,
            #                                    bool populate_best_linearization = false);
            self.assertIsInstance(opt.optimize(values=values), cc_sym.OptimizationStats)
            self.assertAlmostEqual(values.at(pi_key), math.pi)

            self.assertIsInstance(
                opt.optimize(values=values, num_iterations=2), cc_sym.OptimizationStats
            )
            self.assertIsInstance(
                opt.optimize(values=values, populate_best_linearization=True),
                cc_sym.OptimizationStats,
            )
            self.assertIsInstance(opt.optimize(values, 2, True), cc_sym.OptimizationStats)

            # Testing the wrapping of overload
            # void Optimize(Values<Scalar>* values, int num_iterations,
            #               bool populate_best_linearization, OptimizationStats<Scalar>* stats);
            self.assertIsNone(
                opt.optimize(
                    values=values,
                    num_iterations=2,
                    populate_best_linearization=False,
                    stats=cc_sym.OptimizationStats(),
                )
            )
            self.assertIsNone(opt.optimize(values, 2, False, cc_sym.OptimizationStats()))

            # Testing the wrapping of overload
            # void Optimize(Values<Scalar>* values, int num_iterations,
            #               OptimizationStats<Scalar>* stats);
            self.assertIsNone(
                opt.optimize(values=values, num_iterations=2, stats=cc_sym.OptimizationStats())
            )
            self.assertIsNone(opt.optimize(values, 2, cc_sym.OptimizationStats()))

            # Testing the wrapping of overlaod
            # void Optimize(Values<Scalar>* values, OptimizationStats<Scalar>* stats);
            self.assertIsNone(opt.optimize(values=values, stats=cc_sym.OptimizationStats()))
            self.assertIsNone(opt.optimize(values, cc_sym.OptimizationStats()))

            # Testing that the passed in stats are actually modified
            stats = cc_sym.OptimizationStats()
            self.assertEqual(len(stats.iterations), 0)
            opt.optimize(values=values, stats=stats)
            self.assertNotEqual(len(stats.iterations), 0)

        with self.subTest(msg="Optimizer.linearize has been wrapped"):
            values = cc_sym.Values()
            values.set(pi_key, 2.0)
            opt = make_opt()
            self.assertIsInstance(opt.linearize(values=values), cc_sym.Linearization)

        with self.subTest(msg="The methods of Linearization have been wrapped"):
            cc_sym.Linearization()

            values = cc_sym.Values()
            values.set(pi_key, 3.0)
            opt = make_opt()
            lin = opt.linearize(values=values)

            lin.residual = lin.residual
            lin.hessian_lower = lin.hessian_lower
            lin.jacobian = lin.jacobian
            lin.rhs = lin.rhs

            lin.set_initialized()
            self.assertTrue(lin.is_initialized())
            lin.reset()
            self.assertFalse(lin.is_initialized())
            lin.set_initialized(initialized=True)
            self.assertTrue(lin.is_initialized())
            lin.set_initialized(initialized=False)
            self.assertFalse(lin.is_initialized())

            lin.set_initialized()
            self.assertIsInstance(lin.error(), T.Scalar)
            self.assertIsInstance(lin.linear_error(x_update=np.array([0.01])), T.Scalar)
            lin.linear_error(np.array([0.01]))

        with self.subTest(msg="cc_sym.Linearization is pickleable"):

            linearization = cc_sym.Linearization()
            linearization.residual = np.array([1, 2, 3])
            linearization.jacobian = sparse.csc_matrix([[1, 2], [3, 4], [5, 6]])
            linearization.hessian_lower = sparse.csc_matrix([[35, 0], [44, 56]])
            linearization.rhs = np.array([22, 28])

            self.assertTrue(
                self.compare_linearizations(
                    linearization, pickle.loads(pickle.dumps(linearization))
                )
            )

            linearization.set_initialized(True)

            self.assertTrue(
                self.compare_linearizations(
                    linearization, pickle.loads(pickle.dumps(linearization))
                )
            )

        with self.subTest(msg="Optimizer.compute_all_covariances has been wrapped"):
            values = cc_sym.Values()
            values.set(pi_key, 2.0)
            opt = make_opt()
            opt.optimize(values=values)
            self.assertIsInstance(
                opt.compute_all_covariances(linearization=opt.linearize(values)), dict
            )

        with self.subTest(msg="Optimizer.compute_covariances has been wrapped"):
            values = cc_sym.Values()
            values.set(pi_key, 1.0)
            opt = make_opt()
            self.assertIsInstance(
                opt.compute_covariances(linearization=opt.linearize(values), keys=[pi_key]), dict
            )

        with self.subTest(msg="Optimzer.keys is wrapped"):
            opt = make_opt()
            self.assertEqual(opt.keys(), [pi_key])

        with self.subTest(msg="Optimizer.update_params is wrapped"):
            opt = make_opt()
            opt.update_params(params=cc_sym.default_optimizer_params())

        with self.subTest(msg="cc_sym.optimize is wrapped"):
            values = cc_sym.Values()
            values.set(pi_key, 3.0)
            cc_sym.optimize(
                params=cc_sym.default_optimizer_params(),
                factors=[pi_factor],
                values=values,
                epsilon=1e-9,
            )
            self.assertAlmostEqual(values.at(pi_key), math.pi)

            cc_sym.optimize(
                params=cc_sym.default_optimizer_params(), factors=[pi_factor], values=values
            )

    def test_default_params_match(self) -> None:
        """
        Check that the default params in C++ and Python are the same

        Except verbose, which defaults to False in C++ and True in Python
        """
        self.assertEqual(
            cc_sym.default_optimizer_params(),
            optimizer_params_t(**dataclasses.asdict(optimizer.Optimizer.Params(verbose=False))),
        )


if __name__ == "__main__":
    TestCase.main()
