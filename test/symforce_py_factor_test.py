import functools
from pathlib import Path
import textwrap

from symforce import geo
from symforce import ops
from symforce import sympy as sm
from symforce import typing as T
from symforce.opt.factor import Factor
from symforce.opt.factor import visualize_factors
from symforce.ops.interfaces import LieGroup
from symforce.test_util import TestCase
from symforce.values import Values


class SymforcePyFactorTest(TestCase):
    """
    Test the Python Factor (`symforce.opt.factor.Factor`).
    """

    @staticmethod
    def create_chain(
        keys: T.Sequence[str], value_type: T.Type[LieGroup], epsilon: T.Scalar = sm.default_epsilon
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

        for i in range(len(keys)):
            x_prior = ops.GroupOps.identity(value_type)
            yield Factor(
                keys=[keys[i]], name="prior", residual=functools.partial(between, y=x_prior)
            )

    def test_basic(self) -> None:
        """
        Test that we can just construct some simple factors.
        """
        num_samples = 5
        xs = [f"x{i}" for i in range(num_samples)]

        factors = list(self.create_chain(keys=xs, value_type=geo.Rot3))

        self.assertEqual(len(factors), 2 * num_samples - 1)

        first_between = factors[0]
        self.assertEqual(set(first_between.keys), {"x0", "x1"})
        self.assertEqual(first_between.name, "between")
        self.assertEqual(
            first_between.codegen.inputs, Values(x=geo.Rot3.symbolic("x"), y=geo.Rot3.symbolic("y"))
        )

    def test_visualize(self) -> None:
        """
        Test the `visualize_factors` method.
        """
        num_samples = 3
        xs = [f"x{i}" for i in range(num_samples)]

        factors = list(self.create_chain(keys=xs, value_type=geo.Rot3))

        dot_graph = visualize_factors(factors)

        expected = (Path(__file__).parent / "test_data" / "py_factor_test.gv").read_text()

        self.maxDiff = None
        self.assertMultiLineEqual(str(dot_graph), expected)


if __name__ == "__main__":
    TestCase.main()
