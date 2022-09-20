# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Shared helper code between codegen of all languages.
"""

from __future__ import annotations

import dataclasses
import importlib.abc
import importlib.util
import itertools
import sys
from pathlib import Path

import sympy

import symforce
import symforce.symbolic as sf
from symforce import _sympy_count_ops
from symforce import ops
from symforce import typing as T
from symforce import typing_util
from symforce.codegen import codegen_config
from symforce.codegen import format_util
from symforce.values import IndexEntry
from symforce.values import Values

NUMPY_DTYPE_FROM_SCALAR_TYPE = {"double": "numpy.float64", "float": "numpy.float32"}
# Type representing generated code (list of lhs and rhs terms)
T_terms = T.Sequence[T.Tuple[sf.Symbol, sf.Expr]]
T_nested_terms = T.Sequence[T_terms]
T_terms_printed = T.Sequence[T.Tuple[str, str]]


class DenseAndSparseOutputTerms(T.NamedTuple):
    dense: T.List[T.List[sf.Expr]]
    sparse: T.List[T.List[sf.Expr]]


class OutputWithTerms(T.NamedTuple):
    name: str
    type: T.Element
    terms: T_terms_printed


class PrintCodeResult(T.NamedTuple):
    intermediate_terms: T_terms_printed
    dense_terms: T.List[OutputWithTerms]
    sparse_terms: T.List[OutputWithTerms]
    total_ops: int


@dataclasses.dataclass
class CSCFormat:
    """
    A matrix written in Compressed Sparse Column format.
    """

    kRows: int  # Number of rows
    kCols: int  # Number of columns
    kNumNonZero: int  # Number of nonzero entries
    kColPtrs: T.List[int]  # nonzero_elements[kColPtrs[col]] is the first nonzero entry of col
    kRowIndices: T.List[int]  # row indices of nonzero entries written in column-major order
    nonzero_elements: T.List[sf.Scalar]  # nonzero entries written in column-major order

    @staticmethod
    def from_matrix(sparse_matrix: sf.Matrix) -> CSCFormat:
        """
        Returns a dictionary with the metadata required to represent a matrix as a sparse matrix
        in CSC form.

        Args:
            sparse_matrix: A symbolic sf.Matrix where sparsity is given by exact zero equality.
        """
        kColPtrs = []
        kRowIndices = []
        nonzero_elements = []
        data_inx = 0
        # Loop through columns because we assume CSC form
        for j in range(sparse_matrix.shape[1]):
            kColPtrs.append(data_inx)
            for i in range(sparse_matrix.shape[0]):
                if sparse_matrix[i, j] == 0:
                    continue
                kRowIndices.append(i)
                nonzero_elements.append(sparse_matrix[i, j])
                data_inx += 1
        kColPtrs.append(data_inx)

        return CSCFormat(
            kRows=sparse_matrix.rows,
            kCols=sparse_matrix.cols,
            kNumNonZero=len(nonzero_elements),
            kColPtrs=kColPtrs,
            kRowIndices=kRowIndices,
            nonzero_elements=nonzero_elements,
        )


def print_code(
    inputs: Values,
    outputs: Values,
    sparse_mat_data: T.Dict[str, CSCFormat],
    config: codegen_config.CodegenConfig,
    cse: bool = True,
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

    Returns:
        T.List[T.Tuple[str, str]]: Line of code per temporary variable
        T.List[OutputWithTerms]: Collection of lines of code per dense output variable
        T.List[OutputWithTerms]: Collection of lines of code per sparse output variable
        int: Total number of ops
    """
    # Split outputs into dense and sparse outputs, since we treat them differently when doing codegen
    dense_outputs = Values()
    sparse_outputs = Values()
    for key, value in outputs.items():
        if key in sparse_mat_data:
            sparse_outputs[key] = sparse_mat_data[key].nonzero_elements
        else:
            dense_outputs[key] = value

    output_exprs = DenseAndSparseOutputTerms(
        dense=[ops.StorageOps.to_storage(value) for key, value in dense_outputs.items()],
        sparse=[ops.StorageOps.to_storage(value) for key, value in sparse_outputs.items()],
    )

    # CSE If needed
    if cse:
        temps, simplified_outputs = perform_cse(
            output_exprs=output_exprs,
            cse_optimizations=config.cse_optimizations,
        )
    else:
        temps = []
        simplified_outputs = output_exprs

    # Replace default symbols with vector notation (e.g. "R_re" -> "_R[0]")
    temps_formatted, dense_outputs_formatted, sparse_outputs_formatted = format_symbols(
        inputs=inputs,
        dense_outputs=dense_outputs,
        sparse_outputs=sparse_outputs,
        intermediate_terms=temps,
        output_terms=simplified_outputs,
        config=config,
    )

    simpify_list = lambda lst: [sympy.S(term) for term in lst]
    simpify_nested_lists = lambda nested_lsts: [simpify_list(lst) for lst in nested_lsts]

    temps_formatted = simpify_list(temps_formatted)
    dense_outputs_formatted = simpify_nested_lists(dense_outputs_formatted)
    sparse_outputs_formatted = simpify_nested_lists(sparse_outputs_formatted)

    def count_ops(expr: T.Any) -> int:
        op_count = _sympy_count_ops.count_ops(expr)
        assert isinstance(op_count, int)
        return op_count

    total_ops = (
        count_ops(temps_formatted)
        + count_ops(dense_outputs_formatted)
        + count_ops(sparse_outputs_formatted)
    )

    # Get printer
    printer = config.printer()

    # Print code
    intermediate_terms = [(str(var), printer.doprint(t)) for var, t in temps_formatted]
    dense_outputs_code_no_names = [
        [(str(var), printer.doprint(t)) for var, t in single_output_terms]
        for single_output_terms in dense_outputs_formatted
    ]
    sparse_outputs_code_no_names = [
        [(str(var), printer.doprint(t)) for var, t in single_output_terms]
        for single_output_terms in sparse_outputs_formatted
    ]

    # Pack names and types with outputs
    dense_terms = [
        OutputWithTerms(key, value, output_code_no_name)
        for output_code_no_name, (key, value) in zip(
            dense_outputs_code_no_names, dense_outputs.items()
        )
    ]
    sparse_terms = [
        OutputWithTerms(key, value, sparse_output_code_no_name)
        for sparse_output_code_no_name, (key, value) in zip(
            sparse_outputs_code_no_names, sparse_outputs.items()
        )
    ]

    return PrintCodeResult(
        intermediate_terms=intermediate_terms,
        dense_terms=dense_terms,
        sparse_terms=sparse_terms,
        total_ops=total_ops,
    )


def perform_cse(
    output_exprs: DenseAndSparseOutputTerms,
    cse_optimizations: T.Union[
        T.Literal["basic"], T.Sequence[T.Tuple[T.Callable, T.Callable]]
    ] = None,
) -> T.Tuple[T_terms, DenseAndSparseOutputTerms]:
    """
    Run common sub-expression elimination on the given input/output values.

    Args:
        output_exprs: expressions on which to perform cse
        cse_optimizations: optimizations to be forwarded to sf.cse

    Returns:
        T_terms: Temporary variables holding the common sub-expressions found within output_exprs
        DenseAndSparseOutputTerms: output_exprs, but in terms of the returned temporaries.
    """
    # Perform CSE
    flat_output_exprs = [
        x for storage in (output_exprs.dense + output_exprs.sparse) for x in storage
    ]

    def tmp_symbols() -> T.Iterable[sf.Symbol]:
        for i in itertools.count():
            yield sf.Symbol(f"_tmp{i}")

    if cse_optimizations is not None:
        if symforce.get_symbolic_api() == "symengine":
            raise ValueError("cse_optimizations is not supported on symengine")

        temps, flat_simplified_outputs = sf.cse(
            flat_output_exprs, symbols=tmp_symbols(), optimizations=cse_optimizations
        )
    else:
        temps, flat_simplified_outputs = sf.cse(flat_output_exprs, symbols=tmp_symbols())

    # Unflatten output of CSE
    simplified_outputs = DenseAndSparseOutputTerms(dense=[], sparse=[])
    flat_i = 0
    for storage in output_exprs.dense:
        simplified_outputs.dense.append(flat_simplified_outputs[flat_i : flat_i + len(storage)])
        flat_i += len(storage)
    for storage in output_exprs.sparse:
        simplified_outputs.sparse.append(flat_simplified_outputs[flat_i : flat_i + len(storage)])
        flat_i += len(storage)

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

    formatted_input_args, original_args = get_formatted_list(inputs, config, format_as_inputs=True)
    input_subs = dict(
        zip(
            itertools.chain.from_iterable(original_args),
            itertools.chain.from_iterable(formatted_input_args),
        )
    )

    intermediate_terms_formatted = list(
        zip(
            (lhs for lhs, _ in intermediate_terms),
            ops.StorageOps.subs(
                [rhs for _, rhs in intermediate_terms], input_subs, dont_flatten_args=True
            ),
        )
    )

    dense_output_lhs_formatted, _ = get_formatted_list(
        dense_outputs, config, format_as_inputs=False
    )
    dense_output_terms_formatted = [
        list(zip(lhs_formatted, subbed_storage))
        for lhs_formatted, subbed_storage in zip(
            dense_output_lhs_formatted,
            ops.StorageOps.subs(output_terms.dense, input_subs, dont_flatten_args=True),
        )
    ]

    sparse_output_lhs_formatted = get_formatted_sparse_list(sparse_outputs)
    sparse_output_terms_formatted = [
        list(zip(lhs_formatted, subbed_storage))
        for lhs_formatted, subbed_storage in zip(
            sparse_output_lhs_formatted,
            ops.StorageOps.subs(output_terms.sparse, input_subs, dont_flatten_args=True),
        )
    ]

    return intermediate_terms_formatted, dense_output_terms_formatted, sparse_output_terms_formatted


def get_formatted_list(
    values: Values, config: codegen_config.CodegenConfig, format_as_inputs: bool
) -> T.Tuple[T.List[T.List[T.Union[sf.Symbol, sf.DataBuffer]]], T.List[T.List[sf.Scalar]]]:
    """
    Returns a nested list of formatted symbols, as well as a nested list of the corresponding
    original scalar values. For use in generated functions.

    Args:
        values: Values object mapping keys to different objects. Here we only
                use the object types, not their actual values.
        config: Programming language and configuration for when language-specific formatting is
                required
        format_as_inputs: True if values defines the input symbols, false if values defines output
                          expressions.
    Returns:
        flattened_formatted_symbolic_values: nested list of formatted scalar symbols
        flattened_original_values: nested list of original scalar values
    """
    flattened_formatted_symbolic_values = []
    flattened_original_values = []
    for key, value in values.items():
        arg_cls = typing_util.get_type(value)
        storage_dim = ops.StorageOps.storage_dim(value)

        # For each item in the given Values object, we construct a list of symbols used
        # to access the scalar elements of the object. These symbols will later be matched up
        # with the flattened Values object symbols.
        if issubclass(arg_cls, sf.DataBuffer):
            formatted_symbols = [sf.DataBuffer(key, value.shape[0])]
            flattened_value = [value]
        elif isinstance(value, (sf.Expr, sf.Symbol)):
            formatted_symbols = [sf.Symbol(key)]
            flattened_value = [value]
        elif issubclass(arg_cls, sf.Matrix):
            # NOTE(brad): The order of the symbols must match the storage order of sf.Matrix
            # (as returned by sf.Matrix.to_storage). Hence, if there storage order were
            # changed to, say, row major, the below for loops would have to be swapped to
            # reflect that.
            formatted_symbols = []
            for j in range(value.shape[1]):
                for i in range(value.shape[0]):
                    formatted_symbols.append(
                        sf.Symbol(config.format_matrix_accessor(key, i, j, shape=value.shape))
                    )

            flattened_value = ops.StorageOps.to_storage(value)

        elif issubclass(arg_cls, Values):
            # Term is a Values object, so we must flatten it. Here we loop over the index so that
            # we can use the same code with lists.
            formatted_symbols = []
            flattened_value = value.to_storage()
            for name, index_value in value.index().items():
                # Elements of a Values object are accessed with the "." operator
                formatted_symbols.extend(
                    _get_scalar_keys_recursive(
                        index_value, prefix=f"{key}.{name}", config=config, use_data=False
                    )
                )

            assert len(formatted_symbols) == len(
                set(formatted_symbols)
            ), "Non-unique keys:\n{}".format(
                [symbol for symbol in formatted_symbols if formatted_symbols.count(symbol) > 1]
            )
        elif issubclass(arg_cls, (list, tuple)):
            # Term is a list, so we loop over the index of the list, i.e.
            # "values.index()[key].item_index".
            formatted_symbols = []
            flattened_value = ops.StorageOps.to_storage(value)

            sub_index = values.index()[key].item_index
            assert sub_index is not None
            for i, sub_index_val in enumerate(sub_index.values()):
                # Elements of a list are accessed with the "[]" operator.
                formatted_symbols.extend(
                    _get_scalar_keys_recursive(
                        sub_index_val,
                        prefix=f"{key}[{i}]",
                        config=config,
                        use_data=format_as_inputs,
                    )
                )

            assert len(formatted_symbols) == len(
                set(formatted_symbols)
            ), "Non-unique keys:\n{}".format(
                [symbol for symbol in formatted_symbols if formatted_symbols.count(symbol) > 1]
            )
        else:
            if format_as_inputs:
                # For readability, we will store the data of geo/cam objects in a temp vector named "_key"
                # where "key" is the name of the given input variable (can be "self" for member functions accessing
                # object data)
                formatted_symbols = [sf.Symbol(f"_{key}[{j}]") for j in range(storage_dim)]
            else:
                # For geo/cam objects being output, we can't access "data" directly, so in the
                # jinja template we will construct a new object from a vector
                formatted_symbols = [sf.Symbol(f"{key}[{j}]") for j in range(storage_dim)]
            flattened_value = ops.StorageOps.to_storage(value)

        flattened_formatted_symbolic_values.append(formatted_symbols)
        flattened_original_values.append(flattened_value)
    return flattened_formatted_symbolic_values, flattened_original_values


def _get_scalar_keys_recursive(
    index_value: IndexEntry, prefix: str, config: codegen_config.CodegenConfig, use_data: bool
) -> T.List[sf.Symbol]:
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
    if issubclass(datatype, sf.Scalar):
        # Element is a scalar, no need to access subvalues
        vec.append(sf.Symbol(prefix))
    elif issubclass(datatype, Values):
        assert index_value.item_index is not None
        # Recursively add subitems using "." to access subvalues
        for name, sub_index_val in index_value.item_index.items():
            vec.extend(
                _get_scalar_keys_recursive(
                    sub_index_val, prefix=f"{prefix}.{name}", config=config, use_data=False
                )
            )
    elif issubclass(datatype, sf.DataBuffer):
        vec.append(sf.DataBuffer(prefix))
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
    elif issubclass(datatype, sf.Matrix) or not use_data:
        if config.use_eigen_types:
            vec.extend(
                sf.Symbol(config.format_eigen_lcm_accessor(prefix, i))
                for i in range(index_value.storage_dim)
            )
        else:
            vec.extend(sf.Symbol(f"{prefix}[{i}]") for i in range(index_value.storage_dim))
    else:
        # We have a geo/cam or other object that uses "data" to store a flat vector of scalars.
        vec.extend(
            sf.Symbol(config.format_data_accessor(prefix=prefix, index=i))
            for i in range(index_value.storage_dim)
        )

    assert len(vec) == len(set(vec)), "Non-unique keys:\n{}".format(
        [symbol for symbol in vec if vec.count(symbol) > 1]
    )

    return vec


def get_formatted_sparse_list(sparse_outputs: Values) -> T.List[T.List[sf.Scalar]]:
    """
    Returns a nested list of symbols for use in generated functions for sparse matrices.
    """
    symbolic_args = []
    # Each element of sparse_outputs is a list of the nonzero terms in the sparse matrix
    for key, sparse_matrix_data in sparse_outputs.items():
        symbolic_args.append(
            [sf.Symbol(f"{key}_value_ptr[{i}]") for i in range(len(sparse_matrix_data))]
        )

    return symbolic_args


def _load_generated_package_internal(name: str, path: Path) -> T.Tuple[T.Any, T.List[str]]:
    """
    Dynamically load generated package (or module).

    Returns the generated package (module) and a list of the names of all modules added
    to sys.module by this function.

    Does not remove the modules it imports from sys.modules.

    Precondition: If m is a module from the same package as name and is imported by name, then
    there does not exist a different module with the same name as m in sys.modules. This is to
    ensure name imports the correct modules.
    """
    if path.is_dir():
        path = path / "__init__.py"

    parts = name.split(".")
    if len(parts) > 1:
        # Load parent packages
        _, added_module_names = _load_generated_package_internal(
            ".".join(parts[:-1]), path.parent / "__init__.py"
        )
    else:
        added_module_names = []

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    added_module_names.append(name)

    # For mypy: https://github.com/python/typeshed/issues/2793
    assert isinstance(spec.loader, importlib.abc.Loader)

    spec.loader.exec_module(module)
    return module, added_module_names


def load_generated_package(name: str, path: T.Openable) -> T.Any:
    """
    Dynamically load generated package (or module).
    """
    # NOTE(brad): We remove all possibly conflicting modules from the cache. This is
    # to ensure that when name is executed, it loads local modules (if any) rather
    # than any with colliding names that have been loaded elsewhere
    root_package_name = name.split(".")[0]
    callee_saved_modules: T.List[T.Tuple[str, T.Any]] = []
    for module_name in tuple(sys.modules.keys()):
        if root_package_name == module_name.split(".")[0]:
            try:
                conflicting_module = sys.modules[module_name]
                del sys.modules[module_name]
                callee_saved_modules.append((module_name, conflicting_module))
            except KeyError:
                pass

    module, added_module_names = _load_generated_package_internal(name, Path(path))

    # We remove the temporarily added modules
    for added_name in added_module_names:
        try:
            del sys.modules[added_name]
        except KeyError:
            pass

    # And we restore the original removed modules
    for removed_name, removed_module in callee_saved_modules:
        sys.modules[removed_name] = removed_module

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
            namespace, "my_type_t", codegen_data.python_types_dir
        )
        my_type_msg = my_type_t(foo=5)

    Args:
        package: The name of the LCM package for the type
        type_name: The name of the LCM type itself (not including the package)
        lcmtypes_path: The path to the directory containing the generated lcmtypes package

    Returns:
        The Python LCM type
    """
    # We need to import the lcmtypes package first so that sys.path is set up correctly, since this
    # is a namespace package
    import lcmtypes  # pylint: disable=unused-import

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
    lcm_type_dir: T.Openable, lcm_files: T.Sequence[str], lcm_output_dir: T.Openable = None
) -> T.Dict[str, Path]:
    """
    Generates the language-specific type files for all symforce generated ".lcm" files.

    Args:
        lcm_type_dir: Directory containing symforce-generated .lcm files
        lcm_files: List of .lcm files to process
    """
    lcm_type_dir = Path(lcm_type_dir)

    if lcm_output_dir is None:
        lcm_output_dir = lcm_type_dir / ".."
    else:
        lcm_output_dir = Path(lcm_output_dir)

    python_types_dir = lcm_output_dir / "python"
    cpp_types_dir = lcm_output_dir / "cpp" / "lcmtypes"
    lcm_include_dir = "lcmtypes"

    result = {"python_types_dir": python_types_dir, "cpp_types_dir": cpp_types_dir}

    # TODO(brad, aaron): Do something reasonable with lcm_files other than returning early
    # If no LCM files provided, do nothing
    if not lcm_files:
        return result

    from skymarshal import skymarshal
    from skymarshal.emit_cpp import SkymarshalCpp
    from skymarshal.emit_python import SkymarshalPython

    skymarshal.main(
        [SkymarshalPython, SkymarshalCpp],
        args=[
            str(lcm_type_dir),
            "--python",
            "--python-path",
            str(python_types_dir / "lcmtypes"),
            "--python-namespace-packages",
            "--python-package-prefix",
            "lcmtypes",
            "--cpp",
            "--cpp-hpath",
            str(cpp_types_dir),
            "--cpp-include",
            lcm_include_dir,
        ],
    )

    # Autoformat generated python files
    format_util.format_py_dir(python_types_dir)

    return result


def flat_symbols_from_values(values: Values) -> T.List[T.Any]:
    """
    Returns a flat list of unique symbols in the object for codegen
    Note that this *does not* respect storage ordering
    """
    symbols_list = values.to_storage()

    for v in values.values_recursive():
        if isinstance(v, sf.DataBuffer):
            symbols_list.append(v)
    return symbols_list
