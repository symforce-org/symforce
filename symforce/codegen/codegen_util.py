"""
Shared helper code between codegen of all languages.
"""

from enum import Enum
import imp
import os
import inspect

from symforce import ops
from symforce import geo
from symforce.values import Values
from symforce import sympy as sm
from symforce import types as T
from symforce.codegen import printers
from symforce import python_util


NUMPY_DTYPE_FROM_SCALAR_TYPE = {"double": "numpy.float64", "float": "numpy.float32"}


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

    # Substitute names of temp symbols
    tmp_name = lambda i: "_tmp{}".format(i)
    temps_renames = [(t[0], sm.Symbol(tmp_name(i))) for i, t in enumerate(temps)]
    temps_renames_dict = dict(temps_renames)
    temps = [
        (temps_renames[i][1], sm.S(t[1]).subs(temps_renames_dict)) for i, t in enumerate(temps)
    ]
    simplified_outputs = [sm.S(v).subs(temps_renames_dict) for v in simplified_outputs]

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


def format_symbols(
    inputs,  # type: Values
    intermediate_terms,  # type: T.Sequence[T.Tuple[T.Scalar, T.Scalar]]
    output_terms,  # type: T.Sequence[T.Scalar]
):
    # type: (...) -> T.Tuple[T.Sequence[T.Tuple[T.Scalar, T.Scalar]], T.Sequence[T.Scalar]]
    """
    Reformats symbolic variables used in intermediate and outputs terms to be uniform
    across object types. E.g. if we have a rotation object with symbols (R_re, R_im) we will replace
    "R_re" with "_R[0]" and "R_im" with "_R[1]".

    NOTE(nathan): This function would need to be specialized for target languages that use a
    different syntax for accessing data in arrays/matrices
    """
    # Rename the symbolic inputs so that they match the code we generate
    symbolic_args = []
    for key, value in inputs.items():
        arg_cls = python_util.get_type(value)
        if arg_cls == sm.Symbol:
            name_str = "{}"
        elif issubclass(arg_cls, geo.Matrix):
            name_str = "{}[{}]"
        else:
            # For a geo type, we extract the .data() with an underscore prefix
            # to keep the argument as the original variable name.
            name_str = "_{}[{}]"

        if arg_cls == Values:
            storage_dim = len(value.to_storage())
        else:
            storage_dim = ops.StorageOps.storage_dim(value)
        symbols = [sm.Symbol(name_str.format(key, j)) for j in range(storage_dim)]
        symbolic_args.extend(symbols)

    input_subs = dict(zip(inputs.to_storage(), symbolic_args))

    intermediate_terms_formatted = [
        (lhs, sm.S(rhs).subs(input_subs)) for lhs, rhs in intermediate_terms
    ]
    output_terms_formatted = [sm.S(out).subs(input_subs) for out in output_terms]

    return intermediate_terms_formatted, output_terms_formatted


def print_code(
    inputs,  # type: Values
    outputs,  # type: Values
    mode,  # type: CodegenMode
    cse=True,  # type: bool
    substitute_inputs=False,  # type: bool
):
    # type: (...) -> T.Tuple[T.Sequence[T.Tuple[str, str]], T.Sequence[str]]
    """
    Return executable code lines from the given input/output values.

    Args:
        inputs: Values object specifying names and symbolic inputs
        outputs: Values object specifying names and output expressions (written in terms
                of the symbolic inputs)
        mode: Programming language in which to generate the expressions
        cse: Perform common sub-expression elimination
        substitute_inputs: If True, replace all input symbols with a uniform array.

    Returns:
        T.Sequence[T.Tuple[str, str]]: Line of code per temporary variable
        T.Sequence[str]: Line of code per output variable
    """
    input_symbols, _ = inputs.flatten()
    output_exprs, _ = outputs.flatten()

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

    # Replace default symbols with vector notation (e.g. "R_re" -> "_R[0]")
    temps_formatted, simplified_outputs_formatted = format_symbols(
        inputs, temps, simplified_outputs
    )

    # Get printer
    printer = get_code_printer(mode)

    # Print code
    temps_code = [(str(var), printer.doprint(t)) for var, t in temps_formatted]
    outputs_code = [printer.doprint(t) for t in simplified_outputs_formatted]

    return temps_code, outputs_code


def get_code_printer(mode):
    # type: (CodegenMode) -> sm.CodePrinter
    """
    Pick a code printer for the given mode.
    """
    # TODO(hayk): Consider symengine printer if this becomes slow.

    if mode in (CodegenMode.PYTHON2, CodegenMode.PYTHON3):
        # Support specifying python2 for different versions of sympy in different ways
        settings = dict()
        if "standard" in printers.PythonCodePrinter._default_settings:
            settings["standard"] = mode.value

        printer = printers.PythonCodePrinter(settings=settings)

        if hasattr(printer, "standard"):
            printer.standard = mode.value

    elif mode == CodegenMode.CPP:
        from sympy.codegen import ast

        printer = printers.CppCodePrinter(
            settings=dict(
                # TODO(hayk): Emit separately for floats and doubles.
                # type_aliases={ast.real: ast.float32}
            )
        )
    else:
        raise NotImplementedError("Unknown codegen mode: {}".format(mode))

    return printer


def load_generated_package(package_dir):
    # type: (str) -> T.Any
    """
    Dynamically load generated package (or module).
    """
    find_data = imp.find_module(os.path.basename(package_dir), [os.path.dirname(package_dir)])

    if find_data is None:
        raise ImportError("Failed to find module: {}".format(package_dir))

    return imp.load_module(os.path.basename(package_dir), *find_data)  # type: ignore


def get_function_argspec(func):
    # type: (T.Callable) -> inspect.ArgSpec
    """
    Python 2 and 3 compatible way to get the argspec of a function using the inspect package.
    """
    try:
        return inspect.getfullargspec(func)  # type: ignore
    except AttributeError:
        return inspect.getargspec(func)
