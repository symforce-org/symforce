"""
Shared helper code between codegen of all languages.
"""

from enum import Enum
import imp
import os

from symforce import sympy as sm
from symforce import types as T


class CodegenMode(Enum):
    """
    Code generation language / style.
    """

    # Python 2 with numpy
    PYTHON2 = "python2"
    # Python 3 with numpy
    PYTHON3 = "python3"
    # C++11 standard
    CPP = "cpp"


def perform_cse(
    input_symbols,  # type: T.Sequence[T.Scalar]
    output_exprs,  # type: T.Sequence[T.Scalar]
    substitute_inputs=True,  # type: bool
):
    # type: (...) -> T.Tuple[T.Sequence[T.Tuple[T.Scalar, T.Scalar]], T.Sequence[T.Scalar]]
    """
    Run common sub-expression elimination on the given input/output values.

    Args:
        input_symbols: AKA inputs.values_recursive()
        output_exprs: AKA outputs.values_recursive()
        substitute_inputs: If True, replace all input symbols with a uniform array.

    Returns:
        T.Sequence[T.Tuple[T.Scalar, T.Scalar]]: Temporary variables and their expressions
        T.Sequence[T.Scalar]: Output expressions based on temporaries
    """
    # Perform CSE
    temps, simplified_outputs = sm.cse(output_exprs)

    def subs(value, *args, **kwargs):
        # type: (T.Scalar, T.Any, T.Any) -> T.Scalar
        """ Substitute function that ignores types with no subs (like literals) """
        try:
            return value.subs(*args, **kwargs)  # type: ignore
        except AttributeError:
            return value

    # Substitute names of temp symbols
    tmp_name = lambda i: "_tmp{}".format(i)
    temps_renames = [(t[0], sm.Symbol(tmp_name(i))) for i, t in enumerate(temps)]
    temps_renames_dict = dict(temps_renames)
    temps = [(temps_renames[i][1], subs(t[1], temps_renames_dict)) for i, t in enumerate(temps)]
    simplified_outputs = [subs(v, temps_renames_dict) for v in simplified_outputs]

    # Substitute symbols to an input array
    # Rather than having the code contain the names of each symbol, we convert the
    # inputs to elements of an array to make generation more uniform.
    if substitute_inputs:
        input_name = lambda i: "inp[{}]".format(i)
        input_array = [sm.Symbol(input_name(i)) for i in range(len(input_symbols))]
        input_subs = dict(zip(input_symbols, input_array))
        temps = [(var, term.subs(input_subs)) for var, term in temps]
        simplified_outputs = [t.subs(input_subs) for t in simplified_outputs]

    return temps, simplified_outputs


def get_code_printer(mode):
    # type: (CodegenMode) -> sm.CodePrinter
    """
    Pick a code printer for the given mode.
    """
    # TODO(hayk): Consider symengine printer if this becomes slow.

    if mode in (CodegenMode.PYTHON2, CodegenMode.PYTHON3):
        from .python.python_code_printer import PythonCodePrinter

        # Support specifying python2 for different versions of sympy in different ways
        settings = dict()
        if "standard" in PythonCodePrinter._default_settings:
            settings["standard"] = mode.value

        printer = PythonCodePrinter(settings=settings)

        if hasattr(printer, "standard"):
            printer.standard = mode.value

    elif mode == CodegenMode.CPP:
        from .cpp.cpp_code_printer import CppCodePrinter
        from sympy.codegen import ast

        printer = CppCodePrinter(
            settings=dict(
                # TODO(hayk): Emit separately for floats and doubles.
                # type_aliases={ast.real: ast.float32}
            )
        )
    else:
        raise NotImplementedError("Unknown codegen mode: {}".format(mode))

    return printer


def print_code(
    input_symbols,  # type: T.Sequence[T.Scalar]
    output_exprs,  # type: T.Sequence[T.Scalar]
    mode,  # type: CodegenMode
    cse=True,  # type: bool
    substitute_inputs=True,  # type: bool
):
    # type: (...) -> T.Tuple[T.Sequence[T.Tuple[str, str]], T.Sequence[str]]
    """
    Return executable code lines from the given input/output values.

    Args:
        input_symbols: AKA inputs.values_recursive()
        output_exprs: AKA outputs.values_recursive()
        mode:
        cse: Perform common sub-expression elimination
        substitute_inputs: If True, replace all input symbols with a uniform array.

    Returns:
        T.Sequence[T.Tuple[str, str]]: Line of code per temporary variable
        T.Sequence[str]: Line of code per output variable
    """
    # CSE If needed
    if cse:
        temps, simplified_outputs = perform_cse(
            input_symbols=input_symbols,
            output_exprs=output_exprs,
            substitute_inputs=substitute_inputs,
        )
    else:
        temps = []
        simplified_outputs = output_exprs

    # Get printer
    printer = get_code_printer(mode)

    # Print code
    temps_code = [(str(var), printer.doprint(t)) for var, t in temps]
    outputs_code = [printer.doprint(t) for t in simplified_outputs]

    return temps_code, outputs_code


def load_generated_package(package_dir):
    # type: (str) -> T.Any
    """
    Dynamically load generated package (or module).
    """
    find_data = imp.find_module(os.path.basename(package_dir), [os.path.dirname(package_dir)])

    if find_data is None:
        raise ImportError("Failed to find module: {}".format(package_dir))

    return imp.load_module(os.path.basename(package_dir), *find_data)  # type: ignore
