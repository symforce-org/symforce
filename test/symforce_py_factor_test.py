# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import functools
import unittest
from pathlib import Path

import numpy as np

import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce import ops
from symforce import typing as T
from symforce.codegen.codegen_util import load_generated_package
from symforce.ops.interfaces import LieGroup
from symforce.opt.factor import Factor
from symforce.opt.factor import visualize_factors
from symforce.opt.numeric_factor import NumericFactor
from symforce.test_util import TestCase
from symforce.values import Values


class SymforcePyFactorTest(TestCase):
    """
    Test the Python Factor (`symforce.opt.factor.Factor` and `symforce.opt.numeric_factor.NumericFactor`).
    """

    @staticmethod
    def create_chain(
        keys: T.Sequence[str], value_type: T.Type[LieGroup], epsilon: sf.Scalar = sf.numeric_epsilon
    ) -> T.Iterator[Factor]:
        """
        Create a factor chain with betweens and priors of the given type.
        """
        ### Between factors

        def between(x: value_type, y: value_type) -> value_type:  # type: ignore
            return x.local_coordinates(y, epsilon=epsilon)  # type: ignore

        for i in range(len(keys) - 1):
            yield Factor(keys=[keys[i], keys[i + 1]], residual=between)

        ### Prior factors

        for key in keys:
            x_prior = ops.GroupOps.identity(value_type)
            yield Factor(keys=[key], name="prior", residual=functools.partial(between, y=x_prior))

    def test_basic(self) -> None:
        """
        Test that we can just construct some simple factors.
        """
        num_samples = 5
        xs = [f"x{i}" for i in range(num_samples)]

        factors = list(self.create_chain(keys=xs, value_type=sf.Rot3))

        self.assertEqual(len(factors), 2 * num_samples - 1)

        first_between = factors[0]
        self.assertEqual(set(first_between.keys), {"x0", "x1"})
        self.assertEqual(first_between.name, "between")
        self.assertEqual(
            first_between.codegen.inputs, Values(x=sf.Rot3.symbolic("x"), y=sf.Rot3.symbolic("y"))
        )

    def test_generate_and_read(self) -> None:
        """
        Tests factor code generation and loading of previously generated factors
        """
        inputs = Values(a=sf.V3.zero(), b=sf.V3.zero())
        optimized_keys = ["a"]

        def between(a: sf.V3, b: sf.V3) -> sf.V3:
            return a - b

        # Generate a new factor
        output_dir = self.make_output_dir("sf_py_factor_test_")
        namespace = "test"
        name = "between_factor"
        numeric_factor = Factor(
            keys=inputs.keys_recursive(), residual=between, name=name
        ).to_numeric_factor(
            optimized_keys=optimized_keys, output_dir=output_dir, namespace=namespace
        )
        # Check that we can call the generated function
        residual, _, _, _ = numeric_factor.linearize(inputs)
        self.assertStorageNear(residual, np.zeros((3,)))

        # Load the generated function
        loaded_factor = NumericFactor.from_file_python(
            keys=inputs.keys_recursive(),
            optimized_keys=optimized_keys,
            output_dir=output_dir,
            namespace=namespace,
            name=name,
        )
        # Check that we can call the loaded function
        residual, _, _, _ = loaded_factor.linearize(inputs)
        self.assertStorageNear(residual, np.zeros((3,)))

    def test_generate_name(self) -> None:
        """
        Tests that the name returned by Factor.generate is actually present in the generated file.
        """

        def residual(x: T.Scalar) -> sf.V1:
            return sf.V1(x)

        for name in ["CamelCase", "snake_case", "Other_Case"]:
            factor = Factor(keys=["x"], residual=residual, name=name)

            output_dir = self.make_output_dir("sf.py_factor_test_")
            metadata = factor.generate(optimized_keys=["x"], output_dir=output_dir)
            function_dir = metadata["function_dir"]

            pkg = load_generated_package(
                f"{function_dir.name}.{name}_factor", function_dir / (name + "_factor.py")
            )

            # NOTE(brad): Will raise error if metadata["name"] is not in pkg
            getattr(pkg, metadata["name"])

    @unittest.expectedFailure
    def test_custom_jacobians(self) -> None:
        """
        Tests factors with custom jacobians

        These currently break the cache: https://github.com/symforce-org/symforce/issues/234

        TODO(aaron): Make this pass
        """
        inputs = Values(a=sf.V3.zero(), b=sf.V3.zero())
        optimized_keys = ["a"]

        def between(a: sf.V3, b: sf.V3) -> sf.V3:
            return a - b

        numeric_factor = Factor(
            keys=inputs.keys_recursive(),
            residual=between,
        ).to_numeric_factor(optimized_keys=optimized_keys)

        residual, jacobian, _, _ = numeric_factor.linearize(inputs)
        self.assertStorageNear(residual, np.zeros((3,)))
        self.assertStorageNear(jacobian, np.eye(3))

        numeric_factor = Factor(
            keys=inputs.keys_recursive(),
            residual=between,
            custom_jacobian_func=lambda args: sf.M33(),
        ).to_numeric_factor(optimized_keys=optimized_keys)

        residual, jacobian, _, _ = numeric_factor.linearize(inputs)
        self.assertStorageNear(residual, np.zeros((3,)))
        self.assertStorageNear(jacobian, np.zeros((3, 3)))

    def test_visualize(self) -> None:
        """
        Test the `visualize_factors` method.
        """
        num_samples = 3
        xs = [f"x{i}" for i in range(num_samples)]

        factors = list(self.create_chain(keys=xs, value_type=sf.Rot3))

        dot_graph = visualize_factors(factors)

        expected = (Path(__file__).parent / "test_data" / "py_factor_test.gv").read_text()

        self.maxDiff = None
        self.assertMultiLineEqual(str(dot_graph), expected)


if __name__ == "__main__":
    TestCase.main()
