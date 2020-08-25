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

# Command used to generate language-specific types from .lcm files
SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__))

USE_SKYMARSHAL = False
LCM_GEN_CMD = os.path.join(SYMFORCE_DIR, "***REMOVED***/bin/lcm-gen")
SKYMARSHAL_CMD = os.path.join(SYMFORCE_DIR, "***REMOVED***/bin/skymarshal")

NUMPY_DTYPE_FROM_SCALAR_TYPE = {"double": "numpy.float64", "float": "numpy.float32"}
# Type representing generated code (list of lhs and rhs terms)
T_terms = T.Sequence[T.Tuple[T.Scalar, T.Scalar]]


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


def print_code(
    inputs,  # type: Values
    outputs,  # type: Values
    sparse_mat_data,  # type: T.Dict[str, T.Dict[str, T.Any]]
    mode,  # type: CodegenMode
    cse=True,  # type: bool
    substitute_inputs=False,  # type: bool
):
    # type: (...) -> T.Tuple[T.List[T.Tuple[str, str]], T.List[T.Tuple[str, str]], T.List[T.Tuple[str, str]]]
    """
    Return executable code lines from the given input/output values.

    Args:
        inputs: Values object specifying names and symbolic inputs
        outputs: Values object specifying names and output expressions (written in terms
            of the symbolic inputs)
        sparse_mat_data: Data used to represent sparse matrices
        mode: Programming language in which to generate the expressions
        sparse_mat_data: Data associated with sparse matrices. sparse_mat_data["keys"] stores
            a list of the keys in outputs which should be treated as sparse matrices
        cse: Perform common sub-expression elimination
        substitute_inputs: If True, replace all input symbols with a uniform array.

    Returns:
        T.Sequence[T.Tuple[str, str]]: Line of code per temporary variable
        T.Sequence[str]: Line of code per output variable
    """
    # Split outputs into dense and sparse outputs, since we treate them differently when doing codegen
    dense_outputs = Values()
    sparse_outputs = Values()
    for key, value in outputs.items():
        if key in sparse_mat_data:
            sparse_outputs[key] = sparse_mat_data[key]["nonzero_elements"]
        else:
            dense_outputs[key] = value

    input_symbols = inputs.to_storage()
    output_exprs = dense_outputs.to_storage() + sparse_outputs.to_storage()

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
    temps_formatted, simplified_outputs_formatted, sparse_terms_formatted = format_symbols(
        inputs=inputs,
        dense_outputs=dense_outputs,
        sparse_outputs=sparse_outputs,
        intermediate_terms=temps,
        output_terms=simplified_outputs,
        mode=mode,
    )

    # Get printer
    printer = get_code_printer(mode)

    # Print code
    temps_code = [(str(var), printer.doprint(t)) for var, t in temps_formatted]
    outputs_code = [(str(var), printer.doprint(t)) for var, t in simplified_outputs_formatted]
    sparse_code = [(str(var), printer.doprint(t)) for var, t in sparse_terms_formatted]

    return temps_code, outputs_code, sparse_code


def perform_cse(
    input_symbols,  # type: T.Sequence[T.Scalar]
    output_exprs,  # type: T.Sequence[T.Scalar]
    substitute_inputs=True,  # type: bool
):
    # type: (...) -> T.Tuple[T_terms, T.List[T.Scalar]]
    """
    Run common sub-expression elimination on the given input/output values.

    Args:
        input_symbols: AKA inputs.values_recursive()
        output_exprs: AKA outputs.values_recursive()
        substitute_inputs: If True, replace all input symbols with a uniform array.

    Returns:
        T_terms: Temporary variables and their expressions
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
    dense_outputs,  # type: Values
    sparse_outputs,  # type: Values
    intermediate_terms,  # type: T_terms
    output_terms,  # type: T.List[T.Scalar]
    mode,  # type: CodegenMode
):
    # type: (...) -> T.Tuple[T_terms, T_terms, T_terms]
    """
    Reformats symbolic variables used in intermediate and outputs terms to match structure of inputs/outputs.
    For example if we have an input array "arr" with symbolic elements [arr0, arr1], we will remap symbol "arr0" to
    "arr[0]" and symbol "arr1" to "arr[1]".
    """
    # Rename the symbolic inputs so that they match the code we generate

    symbolic_args = get_formatted_flat_list(inputs, mode, format_as_inputs=True)
    input_subs = dict(zip(inputs.to_storage(), symbolic_args))
    intermediate_terms_formatted = [
        (lhs, ops.StorageOps.subs(rhs, input_subs)) for lhs, rhs in intermediate_terms
    ]

    dense_output_terms = output_terms[: dense_outputs.storage_dim()]
    sparse_output_terms = output_terms[dense_outputs.storage_dim() :]

    dense_output_lhs_formatted = get_formatted_flat_list(
        dense_outputs, mode, format_as_inputs=False
    )
    dense_output_rhs_formatted = [
        ops.StorageOps.subs(out, input_subs) for out in dense_output_terms
    ]
    assert len(dense_output_lhs_formatted) == len(dense_output_rhs_formatted)
    dense_output_terms_formatted = zip(dense_output_lhs_formatted, dense_output_rhs_formatted)

    sparse_output_lhs_formatted = get_formatted_sparse_flat_list(sparse_outputs, mode)
    sparse_output_rhs_formatted = [
        ops.StorageOps.subs(out, input_subs) for out in sparse_output_terms
    ]
    assert len(sparse_output_lhs_formatted) == len(sparse_output_rhs_formatted)
    sparse_output_terms_formatted = zip(sparse_output_lhs_formatted, sparse_output_rhs_formatted)

    return intermediate_terms_formatted, dense_output_terms_formatted, sparse_output_terms_formatted


def get_formatted_flat_list(values, mode, format_as_inputs):
    # type: (Values, CodegenMode, bool) -> T.List[T.Scalar]
    """
    Returns a flat list of symbols for use in generated functions.

    Args:
        values: Values object mapping keys to different objects. Here we only
                use the object types, not their actual values.
        mode: Target language to use when language-specific formatting is required.
        format_as_inputs: True if values defines the input symbols, false if values defines output expressions.
    """
    symbolic_args = []
    for key, value in values.items():
        arg_cls = python_util.get_type(value)
        storage_dim = ops.StorageOps.storage_dim(value)

        # For each item in the given Values object, we construct a flat list of symbols used
        # to access the scalar elements of the object. These symbols will later be matched up
        # with the flattened Values object symbols.
        symbols = []
        if isinstance(value, (sm.Expr, sm.Symbol)):
            symbols.append(sm.Symbol(key))
        elif issubclass(arg_cls, geo.Matrix):
            if mode == CodegenMode.PYTHON2:
                # TODO(nathan): Not sure this works for 2D matrices
                symbols.extend([sm.Symbol("{}[{}]".format(key, j)) for j in range(storage_dim)])
            elif mode == CodegenMode.CPP:
                for i in range(value.shape[0]):
                    for j in range(value.shape[1]):
                        symbols.append(sm.Symbol("{}({}, {})".format(key, i, j)))
            else:
                raise NotImplementedError()

        elif issubclass(arg_cls, Values):
            # Term is a Values object, so we must flatten it. Here we loop over the index so that
            # we can use the same code with lists.
            vec = []
            for name, index_value in value.index().items():
                # Elements of a Values object are accessed with the "." operator
                vec.extend(
                    _get_scalar_keys_recursive(
                        index_value, prefix="{}.{}".format(key, name), mode=mode, use_data=False
                    )
                )

            assert len(vec) == len(set(vec)), "Non-unique keys:\n{}".format(
                [symbol for symbol in vec if vec.count(symbol) > 1]
            )

            symbols.extend(vec)
        elif issubclass(arg_cls, (list, tuple)):
            # Term is a list, so we loop over the index of the list, i.e. "values.index()[key][3]".
            vec = []
            for i, sub_index_val in enumerate(values.index()[key][3].values()):
                # Elements of a list are accessed with the "[]" operator.
                vec.extend(
                    _get_scalar_keys_recursive(
                        sub_index_val,
                        prefix="{}[{}]".format(key, i),
                        mode=mode,
                        use_data=format_as_inputs,
                    )
                )

            assert len(vec) == len(set(vec)), "Non-unique keys:\n{}".format(
                [symbol for symbol in vec if vec.count(symbol) > 1]
            )

            symbols.extend(vec)
        else:
            if format_as_inputs:
                # For readability, we will store the data of geo/cam objects in a temp vector named "_key"
                # where "key" is the name of the given input variable (can be "self" for member functions accessing
                # object data)
                symbols.extend([sm.Symbol("_{}[{}]".format(key, j)) for j in range(storage_dim)])
            else:
                # For geo/cam objects being output, we can't access "data" directly, so in the
                # jinja template we will construct a new object from a vector
                symbols.extend([sm.Symbol("{}[{}]".format(key, j)) for j in range(storage_dim)])

        symbolic_args.extend(symbols)
    return symbolic_args


def _get_scalar_keys_recursive(index_value, prefix, mode, use_data):
    # type: (T.Any, str, CodegenMode, bool) -> T.List[str]
    """
    Returns a flat vector of keys, recursing on Values or List objects to get sub-elements.

    Args:
        index_value: Entry in a given index consisting of (inx, datatype, shape, item_index)
            See Values.index() for details on how this entry is built.
        prefix: Symbol used to access parent object, e.g. "my_values.item" or "my_list[i]"
        mode: Target language to use when language-specific formatting is required.
        use_data: If true, we assume we can have a list of geo/cam objects whose data can be
            accessed with ".data" or ".Data()". Otherwise, assume geo/cam objects are represented
            by a vector of scalars (e.g. as they are in lcm types).
    """
    # First pull out useful terms from the index entry
    _, datatype, shape, item_index = index_value

    vec = []
    if len(shape) == 0:
        # Element is a scalar, no need to access subvalues
        vec.append(sm.Symbol(prefix))
    elif datatype == "Values":
        # Recursively add subitems using "." to access subvalues
        for name, sub_index_val in item_index.items():
            vec.extend(
                _get_scalar_keys_recursive(
                    sub_index_val, prefix="{}.{}".format(prefix, name), mode=mode, use_data=False
                )
            )
    elif datatype == "List":
        # Assume all elements of list are same type as first element
        # Recursively add subitems using "[]" to access subvalues
        for i, sub_index_val in enumerate(item_index.values()):
            vec.extend(
                _get_scalar_keys_recursive(
                    sub_index_val, prefix="{}[{}]".format(prefix, i), mode=mode, use_data=use_data
                )
            )
    elif datatype == "Matrix" or not use_data:
        # TODO(nathan): I don't think this deals with 2D matrices correctly
        vec.extend(
            sm.Symbol("{}[{}]".format(prefix, i)) for i in range(Values.shape_to_dims(shape))
        )
    else:
        # We have a geo/cam or other object that uses "data" to store a flat vector of scalars.
        if mode == CodegenMode.PYTHON2:
            vec.extend(
                sm.Symbol("{}.data[{}]".format(prefix, i))
                for i in range(Values.shape_to_dims(shape))
            )
        elif mode == CodegenMode.CPP:
            vec.extend(
                sm.Symbol("{}.Data()[{}]".format(prefix, i))
                for i in range(Values.shape_to_dims(shape))
            )
        else:
            raise NotImplementedError()

    assert len(vec) == len(set(vec)), "Non-unique keys:\n{}".format(
        [symbol for symbol in vec if vec.count(symbol) > 1]
    )
    return vec


def get_sparse_mat_data(sparse_matrix):
    # type: (geo.Matrix) -> T.Dict[str, T.Any]
    """
    Returns a dictionary with the metadata required to represent a matrix as a sparse matrix
    in CSC form.

    Args:
        sparse_matrix: A symbolic geo.Matrix where sparsity is given by exact zero equality.
    """
    sparse_mat_data = {}  # type: T.Dict[str, T.Any]
    sparse_mat_data["kNumNonZero"] = 0
    sparse_mat_data["kColPtrs"] = []
    sparse_mat_data["kRowIndices"] = []
    sparse_mat_data["nonzero_elements"] = []
    data_inx = 0
    # Loop through columns because we assume CSC form
    for j in range(sparse_matrix.shape[1]):
        sparse_mat_data["kColPtrs"].append(data_inx)
        for i in range(sparse_matrix.shape[0]):
            if sparse_matrix[i, j] == 0:
                continue
            sparse_mat_data["kNumNonZero"] += 1
            sparse_mat_data["kRowIndices"].append(i)
            sparse_mat_data["nonzero_elements"].append(sparse_matrix[i, j])
            data_inx += 1
    sparse_mat_data["kColPtrs"].append(data_inx)

    return sparse_mat_data


def get_formatted_sparse_flat_list(sparse_outputs, mode):
    # type: (Values, CodegenMode) -> T.List[T.Scalar]
    """
    Returns a flat list of symbols for use in generated functions for sparse matrices.
    """
    symbolic_args = []
    # Each element of sparse_outputs is a list of the nonzero terms in the sparse matrix
    for key, sparse_matrix_data in sparse_outputs.items():
        for i in range(len(sparse_matrix_data)):
            symbolic_args.append(sm.Symbol("{}_value_ptr[{}]".format(key, i)))

    return symbolic_args


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


def get_base_instance(obj):
    # type: (T.Sequence[T.Any]) -> T.Any
    """
    Returns an instance of the base element (e.g. Scalar, Values, Matrix, etc.) of an object.
    If input is a list (incl. multidimensional lists), we return an instance of one of the base
    elements (i.e. the first element that isn't a list). If input is a list we assume all
    elements are of the same type/shape.
    """
    if isinstance(obj, (list, tuple)):
        return get_base_instance(obj[0])
    return obj


def generate_lcm_types(
    lcm_type_dir,  # type: str
    typenames,  # type: T.Sequence[str]
):
    # type: (...) -> T.Dict[str, T.Any]
    """
    Generates the language-specific type files for all symforce generated ".lcm" files.

    Args:
        lcm_type_dir: Directory containing symforce-generated .lcm files
        typenames: List of typenames defined by .lcm files. External types will be ignored.
    """
    python_types_dir = os.path.join(lcm_type_dir, "..", "python2.7", "lcmtypes")
    cpp_types_dir = os.path.join(lcm_type_dir, "..", "cpp", "lcmtypes")
    lcm_include_dir = os.path.join("lcmtypes")

    for name in typenames:
        if "." in name:
            # External type, skip
            continue

        lcm_file = os.path.join(lcm_type_dir, "{}.lcm".format(name))
        if USE_SKYMARSHAL:
            python_util.execute_subprocess(
                [
                    SKYMARSHAL_CMD,
                    lcm_file,
                    "--python",
                    "--python-path",
                    python_types_dir,
                    "--python-empty-init",
                ]
            )
            python_util.execute_subprocess(
                [
                    SKYMARSHAL_CMD,
                    lcm_file,
                    "--cpp",
                    "--cpp-hpath",
                    cpp_types_dir,
                    "--cpp-include",
                    lcm_include_dir,
                ]
            )
        else:
            python_util.execute_subprocess(
                [LCM_GEN_CMD, lcm_file, "--python", "--ppath", python_types_dir]
            )
            python_util.execute_subprocess(
                [
                    LCM_GEN_CMD,
                    lcm_file,
                    "--cpp",
                    "--cpp-hpath",
                    cpp_types_dir,
                    "--cpp-include",
                    lcm_include_dir,
                ]
            )

    return {"python_types_dir": python_types_dir, "cpp_types_dir": cpp_types_dir}
