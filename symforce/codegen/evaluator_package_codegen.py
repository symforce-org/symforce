# mypy: disallow-untyped-defs

import tempfile

from symforce import logger
from symforce import types as T
from symforce.values import Values
from .codegen_util import CodegenMode
from .evaluator_package_common import EvaluatorCodegenSpec
from . import codegen_util


class EvaluatorCodegen(object):
    """
    Generates code to evaluate a symbolic expression specified by input and output values.

    TODO(hayk): Should this just be a function?
    """

    def __init__(self, inputs, outputs, name="codegen"):
        # type: (Values, Values, str) -> None
        """
        Create from input and output values.
        """
        self.name = name

        assert isinstance(inputs, Values)
        self.inputs = inputs

        assert isinstance(outputs, Values)
        self.outputs = outputs

    def generate(
        self,
        mode=CodegenMode.PYTHON2,  # type: CodegenMode
        scalar_type="double",  # type: str
        output_dir=None,  # type: T.Optional[str]
    ):
        # type: (...) -> T.Dict[str, T.Any]
        """
        Generate executable code for the expression this represents.
        """
        # Create output directory if needed
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="sf_codegen_{}_".format(self.name), dir="/tmp")
            logger.debug("Creating temp directory: {}".format(output_dir))

        codegen_spec = EvaluatorCodegenSpec(
            name=self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            mode=mode,
            scalar_type=scalar_type,
            output_dir=output_dir,
        )

        if mode == CodegenMode.PYTHON2:
            from .python import python_evaluator_package

            codegen_data = python_evaluator_package.generate_evaluator(codegen_spec)

            # For python, hot load generated package
            codegen_data["evaluator"] = codegen_util.load_generated_package(
                codegen_data["package_dir"]
            ).Evaluator()
        elif mode == CodegenMode.CPP:
            from .cpp import cpp_evaluator_package

            codegen_data = cpp_evaluator_package.generate_evaluator(codegen_spec)
        else:
            raise NotImplementedError('Unknown mode: "{}"'.format(mode))

        codegen_data["output_dir"] = output_dir
        return codegen_data
