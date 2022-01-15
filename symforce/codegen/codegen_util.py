"""
Shared helper code between codegen of all languages.
"""

import importlib.abc
import importlib.util
import itertools
import os
from pathlib import Path
import sys

from symforce import ops
from symforce import geo
from symforce.values import Values, IndexEntry
from symforce import sympy as sm
from symforce import typing as T
from symforce.codegen import printers, format_util
from symforce.codegen import codegen_config
from symforce import python_util

NUMPY_DTYPE_FROM_SCALAR_TYPE = {"double": "numpy.float64", "float": "numpy.float32"}
# Type representing generated code (list of lhs and rhs terms)
T_terms = T.Sequence[T.Tuple[sm.Symbol, sm.Expr]]
T_nested_terms = T.Sequence[T_terms]
T_terms_printed = T.Sequence[T.Tuple[str, str]]


class DenseAndSparseOutputTerms(T.NamedTuple):
    dense: T.List[T.List[sm.Expr]]
    sparse: T.List[T.List[sm.Expr]]


class OutputWithTerms(T.NamedTuple):
    name: str
    type: T.Element
    terms: T_terms_printed


class PrintCodeResult(T.NamedTuple):
    temps_code: T_terms_printed
    outputs_code: T.List[OutputWithTerms]
    sparse_outputs_code: T.List[OutputWithTerms]
    total_ops: int


def print_code(
    inputs: Values,
    outputs: Values,
    sparse_mat_data: T.Dict[str, T.Dict[str, T.Any]],
    config: codegen_config.CodegenConfig,
    cse: bool = True,
    substitute_inputs: bool = False,
) -> PrintCodeResult:
    """
    Return executable code lines from the given input/output values.

    Args:
        inputs: Values object specifying names and symbolic inputs
        outputs: Values object specifying names and output expressions (written in terms
            of the symbolic inputs)
        sparse_mat_data: Data associated with sparse matrices. sparse_mat_data["keys"] stores
            a list of the keys in outputs which should be treated as sparse matrices
        config: Programming language and configuration in which the expressions are to be generated
        cse: Perform common sub-expression elimination
        substitute_inputs: If True, replace all input symbols with a uniform array.

    Returns:
        T.List[T.Tuple[str, str]]: Line of code per temporary variable
        T.List[OutputWithTerms]: Collection of lines of code per dense output variable
        T.List[OutputWithTerms]: Collection of lines of code per sparse output variable
        int: Total number of ops
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
    output_exprs = DenseAndSparseOutputTerms(
        dense=[ops.StorageOps.to_storage(value) for key, value in dense_outputs.items()],
        sparse=[ops.StorageOps.to_storage(value) for key, value in sparse_outputs.items()],
    )

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

    total_ops = (
        sm.count_ops(temps)
        + sm.count_ops(simplified_outputs.dense)
        + sm.count_ops(simplified_outputs.sparse)
    )

    # Replace default symbols with vector notation (e.g. "R_re" -> "_R[0]")
    temps_formatted, simplified_outputs_formatted, sparse_terms_formatted = format_symbols(
        inputs=inputs,
        dense_outputs=dense_outputs,
        sparse_outputs=sparse_outputs,
        intermediate_terms=temps,
        output_terms=simplified_outputs,
        config=config,
    )

    # Get printer
    printer = get_code_printer(config)

    # Print code
    temps_code = [(str(var), printer.doprint(t)) for var, t in temps_formatted]
    outputs_code_no_names = [
        [(str(var), printer.doprint(t)) for var, t in single_output_terms]
        for single_output_terms in simplified_outputs_formatted
    ]
    sparse_outputs_code_no_names = [
        [(str(var), printer.doprint(t)) for var, t in single_output_terms]
        for single_output_terms in sparse_terms_formatted
    ]

    # Pack names and types with outputs
    outputs_code = [
        OutputWithTerms(key, value, output_code_no_name)
        for output_code_no_name, (key, value) in zip(outputs_code_no_names, dense_outputs.items())
    ]
    sparse_outputs_code = [
        OutputWithTerms(key, value, sparse_output_code_no_name)
        for sparse_output_code_no_name, (key, value) in zip(
            sparse_outputs_code_no_names, sparse_outputs.items()
        )
    ]

    return PrintCodeResult(
        temps_code=temps_code,
        outputs_code=outputs_code,
        sparse_outputs_code=sparse_outputs_code,
        total_ops=total_ops,
    )


def perform_cse(
    input_symbols: T.Sequence[T.Scalar],
    output_exprs: DenseAndSparseOutputTerms,
    substitute_inputs: bool = True,
) -> T.Tuple[T_terms, DenseAndSparseOutputTerms]:
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
    flat_output_exprs = [
        x for storage in (output_exprs.dense + output_exprs.sparse) for x in storage
    ]
    temps, flat_simplified_outputs = sm.cse(flat_output_exprs)

    # Unflatten output of CSE
    simplified_outputs = DenseAndSparseOutputTerms(dense=[], sparse=[])
    flat_i = 0
    for storage in output_exprs.dense:
        simplified_outputs.dense.append(flat_simplified_outputs[flat_i : flat_i + len(storage)])
        flat_i += len(storage)
    for storage in output_exprs.sparse:
        simplified_outputs.sparse.append(flat_simplified_outputs[flat_i : flat_i + len(storage)])
        flat_i += len(storage)

    # Substitute names of temp symbols
    tmp_name = lambda i: f"_tmp{i}"
    temps_renames = [(t[0], sm.Symbol(tmp_name(i))) for i, t in enumerate(temps)]
    temps_renames_dict = dict(temps_renames)
    temps = [
        (temps_renames[i][1], sm.S(t[1]).subs(temps_renames_dict)) for i, t in enumerate(temps)
    ]
    simplified_outputs = DenseAndSparseOutputTerms(
        dense=[
            [sm.S(v).subs(temps_renames_dict) for v in storage]
            for storage in simplified_outputs.dense
        ],
        sparse=[
            [sm.S(v).subs(temps_renames_dict) for v in storage]
            for storage in simplified_outputs.sparse
        ],
    )

    # Substitute symbols to an input array
    # Rather than having the code contain the names of each symbol, we convert the
    # inputs to elements of an array to make generation more uniform.
    if substitute_inputs:
        input_name = lambda i: f"inp[{i}]"
        input_array = [sm.Symbol(input_name(i)) for i in range(len(input_symbols))]
        input_subs = dict(zip(input_symbols, input_array))
        temps = [(var, term.subs(input_subs)) for var, term in temps]
        simplified_outputs = DenseAndSparseOutputTerms(
            dense=[[t.subs(input_subs) for t in storage] for storage in simplified_outputs.dense],
            sparse=[[t.subs(input_subs) for t in storage] for storage in simplified_outputs.sparse],
        )

    return temps, simplified_outputs


def format_symbols(
    inputs: Values,
    dense_outputs: Values,
    sparse_outputs: Values,
    intermediate_terms: T_terms,
    output_terms: DenseAndSparseOutputTerms,
    config: codegen_config.CodegenConfig,
) -> T.Tuple[T_terms, T_nested_terms, T_nested_terms]:
    """
    Reformats symbolic variables used in intermediate and outputs terms to match structure of
    inputs/outputs. For example if we have an input array "arr" with symbolic elements [arr0, arr1],
    we will remap symbol "arr0" to "arr[0]" and symbol "arr1" to "arr[1]".
    """
    # Rename the symbolic inputs so that they match the code we generate

    symbolic_args = list(
        itertools.chain.from_iterable(get_formatted_list(inputs, config, format_as_inputs=True))
    )
    input_subs = dict(zip(inputs.to_storage(), symbolic_args))
    intermediate_terms_formatted = [
        (lhs, ops.StorageOps.subs(rhs, input_subs)) for lhs, rhs in intermediate_terms
    ]

    dense_output_lhs_formatted = get_formatted_list(dense_outputs, config, format_as_inputs=False)
    dense_output_terms_formatted = [
        list(zip(lhs_formatted, ops.StorageOps.subs(storage, input_subs)))
        for lhs_formatted, storage in zip(dense_output_lhs_formatted, output_terms.dense)
    ]

    sparse_output_lhs_formatted = get_formatted_sparse_list(sparse_outputs)
    sparse_output_terms_formatted = [
        list(zip(lhs_formatted, ops.StorageOps.subs(storage, input_subs)))
        for lhs_formatted, storage in zip(sparse_output_lhs_formatted, output_terms.sparse)
    ]

    return intermediate_terms_formatted, dense_output_terms_formatted, sparse_output_terms_formatted


def get_formatted_list(
    values: Values, config: codegen_config.CodegenConfig, format_as_inputs: bool
) -> T.List[T.List[T.Scalar]]:
    """
    Returns a nested list of symbols for use in generated functions.

    Args:
        values: Values object mapping keys to different objects. Here we only
                use the object types, not their actual values.
        config: Programming language and configuration for when language-specific formatting is
                required
        format_as_inputs: True if values defines the input symbols, false if values defines output
                          expressions.
    """
    symbolic_args = []
    for key, value in values.items():
        arg_cls = python_util.get_type(value)
        storage_dim = ops.StorageOps.storage_dim(value)

        # For each item in the given Values object, we construct a list of symbols used
        # to access the scalar elements of the object. These symbols will later be matched up
        # with the flattened Values object symbols.
        if isinstance(value, (sm.Expr, sm.Symbol)):
            symbols = [sm.Symbol(key)]
        elif issubclass(arg_cls, geo.Matrix):
            if isinstance(config, codegen_config.PythonConfig):
                # TODO(nathan): Not sure this works for 2D matrices
                symbols = [sm.Symbol(f"{key}[{j}]") for j in range(storage_dim)]
            elif isinstance(config, codegen_config.CppConfig):
                symbols = []
                for i in range(value.shape[0]):
                    for j in range(value.shape[1]):
                        symbols.append(sm.Symbol(f"{key}({i}, {j})"))
            else:
                raise NotImplementedError()

        elif issubclass(arg_cls, Values):
            # Term is a Values object, so we must flatten it. Here we loop over the index so that
            # we can use the same code with lists.
            symbols = []
            for name, index_value in value.index().items():
                # Elements of a Values object are accessed with the "." operator
                symbols.extend(
                    _get_scalar_keys_recursive(
                        index_value, prefix=f"{key}.{name}", config=config, use_data=False
                    )
                )

            assert len(symbols) == len(set(symbols)), "Non-unique keys:\n{}".format(
                [symbol for symbol in symbols if symbols.count(symbol) > 1]
            )
        elif issubclass(arg_cls, (list, tuple)):
            # Term is a list, so we loop over the index of the list, i.e.
            # "values.index()[key].item_index".
            symbols = []

            sub_index = values.index()[key].item_index
            assert sub_index is not None
            for i, sub_index_val in enumerate(sub_index.values()):
                # Elements of a list are accessed with the "[]" operator.
                symbols.extend(
                    _get_scalar_keys_recursive(
                        sub_index_val,
                        prefix=f"{key}[{i}]",
                        config=config,
                        use_data=format_as_inputs,
                    )
                )

            assert len(symbols) == len(set(symbols)), "Non-unique keys:\n{}".format(
                [symbol for symbol in symbols if symbols.count(symbol) > 1]
            )
        else:
            if format_as_inputs:
                # For readability, we will store the data of geo/cam objects in a temp vector named "_key"
                # where "key" is the name of the given input variable (can be "self" for member functions accessing
                # object data)
                symbols = [sm.Symbol(f"_{key}[{j}]") for j in range(storage_dim)]
            else:
                # For geo/cam objects being output, we can't access "data" directly, so in the
                # jinja template we will construct a new object from a vector
                symbols = [sm.Symbol(f"{key}[{j}]") for j in range(storage_dim)]

        symbolic_args.append(symbols)
    return symbolic_args


def _get_scalar_keys_recursive(
    index_value: IndexEntry, prefix: str, config: codegen_config.CodegenConfig, use_data: bool,
) -> T.List[str]:
    """
    Returns a vector of keys, recursing on Values or List objects to get sub-elements.

    Args:
        index_value: Entry in a given index consisting of (inx, datatype, shape, item_index)
            See Values.index() for details on how this entry is built.
        prefix: Symbol used to access parent object, e.g. "my_values.item" or "my_list[i]"
        config: Programming language and configuration for when language-specific formatting is
                required
        use_data: If true, we assume we can have a list of geo/cam objects whose data can be
            accessed with ".data" or ".Data()". Otherwise, assume geo/cam objects are represented
            by a vector of scalars (e.g. as they are in lcm types).
    """
    vec = []
    datatype = index_value.datatype()
    if issubclass(datatype, T.Scalar):
        # Element is a scalar, no need to access subvalues
        vec.append(sm.Symbol(prefix))
    elif issubclass(datatype, Values):
        assert index_value.item_index is not None
        # Recursively add subitems using "." to access subvalues
        for name, sub_index_val in index_value.item_index.items():
            vec.extend(
                _get_scalar_keys_recursive(
                    sub_index_val, prefix=f"{prefix}.{name}", config=config, use_data=False
                )
            )
    elif issubclass(datatype, (list, tuple)):
        assert index_value.item_index is not None
        # Assume all elements of list are same type as first element
        # Recursively add subitems using "[]" to access subvalues
        for i, sub_index_val in enumerate(index_value.item_index.values()):
            vec.extend(
                _get_scalar_keys_recursive(
                    sub_index_val, prefix=f"{prefix}[{i}]", config=config, use_data=use_data
                )
            )
    elif issubclass(datatype, geo.Matrix) or not use_data:
        # TODO(nathan): I don't think this deals with 2D matrices correctly
        if isinstance(config, codegen_config.PythonConfig) and config.use_eigen_types:
            vec.extend(sm.Symbol(f"{prefix}.data[{i}]") for i in range(index_value.storage_dim))
        else:
            vec.extend(sm.Symbol(f"{prefix}[{i}]") for i in range(index_value.storage_dim))
    else:
        # We have a geo/cam or other object that uses "data" to store a flat vector of scalars.
        if isinstance(config, codegen_config.PythonConfig):
            vec.extend(sm.Symbol(f"{prefix}.data[{i}]") for i in range(index_value.storage_dim))
        elif isinstance(config, codegen_config.CppConfig):
            vec.extend(sm.Symbol(f"{prefix}.Data()[{i}]") for i in range(index_value.storage_dim))
        else:
            raise NotImplementedError()

    assert len(vec) == len(set(vec)), "Non-unique keys:\n{}".format(
        [symbol for symbol in vec if vec.count(symbol) > 1]
    )

    return vec


def get_sparse_mat_data(sparse_matrix: geo.Matrix) -> T.Dict[str, T.Any]:
    """
    Returns a dictionary with the metadata required to represent a matrix as a sparse matrix
    in CSC form.

    Args:
        sparse_matrix: A symbolic geo.Matrix where sparsity is given by exact zero equality.
    """
    sparse_mat_data: T.Dict[str, T.Any] = {}
    sparse_mat_data["kRows"] = sparse_matrix.rows
    sparse_mat_data["kCols"] = sparse_matrix.cols
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


def get_formatted_sparse_list(sparse_outputs: Values) -> T.List[T.List[T.Scalar]]:
    """
    Returns a nested list of symbols for use in generated functions for sparse matrices.
    """
    symbolic_args = []
    # Each element of sparse_outputs is a list of the nonzero terms in the sparse matrix
    for key, sparse_matrix_data in sparse_outputs.items():
        symbolic_args.append(
            [sm.Symbol(f"{key}_value_ptr[{i}]") for i in range(len(sparse_matrix_data))]
        )

    return symbolic_args


def get_code_printer(config: codegen_config.CodegenConfig) -> "sm.CodePrinter":
    """
    Pick a code printer for the given mode.
    """
    # TODO(hayk): Consider symengine printer if this becomes slow.

    if isinstance(config, codegen_config.PythonConfig):
        # Support specifying python2 for different versions of sympy in different ways
        settings = {}
        if (
            "standard"
            in printers.PythonCodePrinter._default_settings  # pylint: disable=protected-access
        ):
            settings["standard"] = config.standard.value

        printer = printers.PythonCodePrinter(settings=settings)

        if hasattr(printer, "standard"):
            printer.standard = config.standard.value

    elif isinstance(config, codegen_config.CppConfig):
        printer = printers.CppCodePrinter(
            settings={
                # TODO(hayk): Emit separately for floats and doubles.
                # type_aliases: {sympy.codegen.ast.real: sympy.codegen.ast.float32}
            }
        )
    else:
        raise NotImplementedError(f"Unknown config type: {config}")

    return printer


def load_generated_package(name: str, path: T.Union[str, Path]) -> T.Any:
    """
    Dynamically load generated package (or module).
    """
    path = Path(path)

    if path.is_dir():
        path = path / "__init__.py"

    parts = name.split(".")
    if len(parts) > 1:
        # Load parent packages
        load_generated_package(".".join(parts[:-1]), path.parent / "__init__.py")

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module

    # For mypy: https://github.com/python/typeshed/issues/2793
    assert isinstance(spec.loader, importlib.abc.Loader)

    spec.loader.exec_module(module)
    return module


def load_generated_lcmtype(
    package: str, type_name: str, lcmtypes_path: T.Union[str, Path]
) -> T.Type:
    """
    Load an LCM type generated by Codegen.generate_function

    Example usage:

        my_codegen = Codegen(my_func, config=PythonConfig())
        codegen_data = my_codegen.generate_function(output_dir=output_dir, namespace=namespace)
        my_type_t = codegen_util.load_generated_lcmtype(
            namespace, "my_type_t", codegen_data["python_types_dir"]
        )
        my_type_msg = my_type_t(foo=5)

    Args:
        package: The name of the LCM package for the type
        type_name: The name of the LCM type itself (not including the package)
        lcmtypes_path: The path to the directory containing the generated lcmtypes package

    Returns:
        The Python LCM type
    """
    return getattr(
        load_generated_package(
            f"lcmtypes.{package}._{type_name}",
            Path(lcmtypes_path) / "lcmtypes" / package / f"_{type_name}.py",
        ),
        type_name,
    )


def get_base_instance(obj: T.Sequence[T.Any]) -> T.Any:
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
    lcm_type_dir: str, lcm_files: T.Sequence[str], lcm_output_dir: T.Optional[str] = None,
) -> T.Dict[str, T.Any]:
    """
    Generates the language-specific type files for all symforce generated ".lcm" files.

    Args:
        lcm_type_dir: Directory containing symforce-generated .lcm files
        lcm_files: List of .lcm files to process
    """
    if lcm_output_dir is None:
        lcm_output_dir = os.path.join(lcm_type_dir, "..")

    python_types_dir = os.path.join(lcm_output_dir, "python2.7")
    cpp_types_dir = os.path.join(lcm_output_dir, "cpp", "lcmtypes")
    lcm_include_dir = os.path.join("lcmtypes")

    result = {"python_types_dir": python_types_dir, "cpp_types_dir": cpp_types_dir}

    # If no LCM files provided, do nothing
    if not lcm_files:
        return result

    from skymarshal import skymarshal
    from skymarshal.emit_python import SkymarshalPython
    from skymarshal.emit_cpp import SkymarshalCpp

    skymarshal.main(
        [SkymarshalPython, SkymarshalCpp],
        args=[
            lcm_type_dir,
            "--python",
            "--python-path",
            os.path.join(python_types_dir, "lcmtypes"),
            "--python-namespace-packages",
            "--python-package-prefix",
            "lcmtypes",
            "--cpp",
            "--cpp-hpath",
            cpp_types_dir,
            "--cpp-include",
            lcm_include_dir,
        ],
    )

    # Autoformat generated python files
    format_util.format_py_dir(python_types_dir)

    return result
