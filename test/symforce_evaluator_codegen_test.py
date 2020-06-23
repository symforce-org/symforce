import logging
import os

from symforce import geo
from symforce import logger
from symforce import python_util
from symforce import sympy as sm
from symforce import types as T
from symforce.codegen import EvaluatorCodegen
from symforce.codegen import CodegenMode
from symforce.test_util import TestCase
from symforce.values import Values


class SymforceEvaluatorCodegenTest(TestCase):
    """
    Test symforce.codegen.EvaluatorCodegen.
    """

    @staticmethod
    def build_values():
        # type: () -> T.Tuple[Values, Values]
        """
        Create some example input/output values.
        """
        inputs = Values()
        x, y = sm.symbols("x y")
        inputs.add(x)
        inputs.add(y)

        inputs["rot"] = geo.Rot3().symbolic("rot")

        # Scalar
        inputs.add(sm.Symbol("constants.epsilon"))

        with inputs.scope("states"):
            # Array element, turns into std::array
            inputs["p"] = geo.V2.symbolic("p")

            # Vector element, turns into Eigen::Vector
            # inputs.add(sm.Symbol('q(0)'))

        outputs = Values()
        outputs["foo"] = x ** 2 + inputs["rot"].q.w
        outputs["bar"] = inputs.attr.constants.epsilon + sm.sin(inputs.attr.y) + x ** 2

        return inputs, outputs

    # -------------------------------------------------------------------------
    # Python
    # -------------------------------------------------------------------------

    def run_python_evaluator(self, evaluator):
        # type: (T.Type) -> None
        """
        Execute and check results of a live loaded python evaluator instance.

        Args:
            evaluator (Evaluator):
        """
        inp = evaluator.input_t()
        inp.constants.epsilon = 1e-8
        inp.x = 1.0
        inp.y = -5.0

        out = evaluator.execute(inp)

        # Check results
        self.assertNear(out.foo, inp.x ** 2 + inp.rot[3])
        self.assertNear(out.bar, inp.constants.epsilon + sm.sin(inp.y) + inp.x ** 2)

    def test_codegen_python(self):
        # type: () -> None
        """
        Test python code generation.
        """
        inputs, outputs = self.build_values()

        for scalar_type in ("double", "float"):
            codegen = EvaluatorCodegen(inputs, outputs, "codegen_test_python")
            codegen_data = codegen.generate(mode=CodegenMode.PYTHON2, scalar_type="float")

            # Run generated example script
            python_util.execute_subprocess(
                ["python", os.path.join(codegen_data["package_dir"], "example", "example.py")]
            )

            self.run_python_evaluator(codegen_data["evaluator"])

            if self.verbose:
                logger.info(codegen_data["package_dir"])

            # Clean up
            if logger.level != logging.DEBUG:
                python_util.remove_if_exists(codegen_data["output_dir"])

    # -------------------------------------------------------------------------
    # C++
    # -------------------------------------------------------------------------

    def run_cpp_evaluator(self, package_dir):
        # type: (str) -> None
        """
        Execute and check results of a generated C++ evaluator package.

        Args:
            package_dir (str):
        """
        # Build example
        example_dir = os.path.join(package_dir, "example")
        make_cmd = ["make", "-C", example_dir]
        if logger.level != logging.DEBUG:
            make_cmd.append("--quiet")
        python_util.execute_subprocess(make_cmd)

        # Run example
        python_util.execute_subprocess(os.path.join(example_dir, "example"))

    def test_codegen_cpp(self):
        # type: () -> None
        """
        Test C++ code generation.
        """
        inputs, outputs = self.build_values()

        for scalar_type in ("double", "float"):
            codegen = EvaluatorCodegen(inputs, outputs, "codegen_test_cpp")
            codegen_data = codegen.generate(mode=CodegenMode.CPP, scalar_type=scalar_type)

            self.run_cpp_evaluator(codegen_data["package_dir"])

            # Clean up
            if logger.level != logging.DEBUG:
                python_util.remove_if_exists(codegen_data["output_dir"])


if __name__ == "__main__":
    TestCase.main()
