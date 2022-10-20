# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import dataclasses
import os
from pathlib import Path

import numpy as np
import scipy.io

import symforce.symbolic as sf
from symforce import codegen
from symforce import logger
from symforce import python_util
from symforce import typing as T
from symforce.codegen import template_util
from symforce.test_util.random_expressions import op_probabilities
from symforce.test_util.random_expressions import unary_binary_expression_gen
from symforce.values import Values

# Parameters controlling the randomly generated expressions
OPS_PER_ENTRY = 5  # Target of ops in the expression tree for each nonzero scalar
N_SYMBOLS = 5  # Number of leaf symbols that the expressions should be functions of

# Number of matrix sparsity patterns to generate functions for
N_MATRICES = 6


def get_matrices() -> T.List[T.Tuple[str, Path, scipy.sparse.csr_matrix]]:
    """
    Load the matrices in the `matrices` folder from disk, and return their names, paths to their
    matrix-market files, and the matrices themselves

    Filters to the first N_MATRICES results, sorted by number of nonzeros
    """
    matrices = []

    for path in (Path(__file__).parent / "matrices").iterdir():
        if not path.is_dir():
            continue

        matrix_name = path.name
        for filename in filter(lambda p: p.suffix == ".mtx", path.iterdir()):
            matrix = scipy.io.mmread(os.fspath(filename))
            if matrix.shape[0] > 1 and matrix.shape[1] > 1:
                break
        else:
            raise FileNotFoundError(f"Didn't find non-vector Matrix Market file in {path}")
        matrices.append(
            (
                matrix_name.replace(" ", "").replace("-", "_"),
                filename,  # pylint: disable=undefined-loop-variable
                matrix.tocsr(),
            )
        )

    matrices.sort(key=lambda m: m[2].nnz)

    return matrices if N_MATRICES is None else matrices[:N_MATRICES]


def _make_return_dynamic(generated_file: Path, shape: T.Tuple[int, int]) -> None:
    """
    Modify a generated file to make the return type of the function an Eigen::MatrixX<Scalar>
    instead of a fixed-size matrix
    """
    generated_file.write_text(
        generated_file.read_text()
        .replace(
            f"Eigen::Matrix<Scalar, {shape[0]}, {shape[1]}>",
            "Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>",
        )
        .replace("> _result;", f"> _result({shape[0]}, {shape[1]});")
    )


def generate_matrix(
    output_dir: Path,
    matrix_name: str,
    matrix: scipy.sparse.csr_matrix,
    symforce_result_is_sparse: bool,
    i: int,
) -> None:
    """
    Generate functions for the given matrix sparsity pattern to compute A, B, and A^T B, in sparse
    and dense forms
    """
    symbols = sf.symbols(f"x:{N_SYMBOLS}")
    gen = unary_binary_expression_gen.UnaryBinaryExpressionGen(
        unary_ops=[op_probabilities.OpProbability("neg", lambda x: -x, 3)],
        binary_ops=[
            op_probabilities.OpProbability("add", lambda x, y: x + y, 4),
            op_probabilities.OpProbability("sub", lambda x, y: x - y, 2),
            op_probabilities.OpProbability("mul", lambda x, y: x * y, 5),
            op_probabilities.OpProbability("div", lambda x, y: x / 2 if y == 0 else x / y, 1),
        ],
        leaves=[-2, -1, 1, 2] + list(symbols),
    )

    def compute_A(*symbols: T.List[sf.Symbol]) -> sf.Matrix:
        exprs = gen.build_expr_vec(OPS_PER_ENTRY * matrix.nnz, matrix.nnz).to_flat_list()
        result = sf.Matrix(*matrix.shape)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] != 0:
                    result[i, j] = exprs.pop()
        return result

    A = compute_A(*symbols)
    B = compute_A(*symbols)
    ATB = A.T * B

    from sympy.simplify import cse_opts

    # These files are large enough that autoformatting them is very slow, so just don't do it
    config = codegen.CppConfig(
        cse_optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)], autoformat=False
    )
    config_noinline = dataclasses.replace(config, force_no_inline=True)

    data = codegen.Codegen(
        inputs=Values(**{s.name: s for s in symbols}),
        outputs=Values(result=A),
        config=config,
        name=f"compute_a_{matrix_name}",
        sparse_matrices=["result"],
        return_key="result",
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)

    data = codegen.Codegen(
        inputs=Values(**{s.name: s for s in symbols}),
        outputs=Values(result=B),
        config=config,
        name=f"compute_b_{matrix_name}",
        sparse_matrices=["result"],
        return_key="result",
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)

    compute_a_dense = codegen.Codegen(
        inputs=Values(**{s.name: s for s in symbols}),
        outputs=Values(result=A),
        config=config,
        name=f"compute_a_dense_{matrix_name}",
        return_key="result",
    )

    data = compute_a_dense.generate_function(output_dir=output_dir, skip_directory_nesting=True)

    assert compute_a_dense.name is not None
    compute_a_dense.name = compute_a_dense.name.replace("_dense_", "_dense_dynamic_")
    data = compute_a_dense.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    _make_return_dynamic(data.generated_files[0], matrix.shape)

    compute_b_dense = codegen.Codegen(
        inputs=Values(**{s.name: s for s in symbols}),
        outputs=Values(result=B),
        config=config,
        name=f"compute_b_dense_{matrix_name}",
        return_key="result",
    )

    data = compute_b_dense.generate_function(output_dir=output_dir, skip_directory_nesting=True)

    assert compute_b_dense.name is not None
    compute_b_dense.name = compute_b_dense.name.replace("_dense_", "_dense_dynamic_")
    data = compute_b_dense.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    _make_return_dynamic(data.generated_files[0], matrix.shape)

    data = codegen.Codegen(
        inputs=Values(**{s.name: s for s in symbols}),
        outputs=Values(result=ATB),
        config=config_noinline,
        name=f"compute_at_b_{matrix_name}",
        return_key="result",
        sparse_matrices=["result"] if symforce_result_is_sparse else [],
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)

    # By default, Eigen will not allocate more than 128KB on the stack
    # https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html
    EIGEN_STACK_ALLOCATION_LIMIT_BYTES = 128 * 2 ** 10

    cant_allocate_on_stack = (
        matrix.shape[0] * matrix.shape[1] * 8 > EIGEN_STACK_ALLOCATION_LIMIT_BYTES
        or matrix.shape[0] * matrix.shape[0] * 8 > EIGEN_STACK_ALLOCATION_LIMIT_BYTES
    )

    if cant_allocate_on_stack and not symforce_result_is_sparse:
        _make_return_dynamic(data.generated_files[0], matrix.shape)

    template_util.render_template(
        template_dir=Path(__file__).parent,
        template_path="matrix_multiplication_benchmark.cc.jinja",
        data=dict(
            matrix_name=matrix_name,
            matrix_name_camel=python_util.snakecase_to_camelcase(matrix_name),
            N=matrix.shape[0],
            M=matrix.shape[1],
            n_runs_multiplier=[100.0, 100.0, 100.0, 10.0, 10.0, 10.0, 1.0, 1.0][i],
            symforce_result_is_sparse=symforce_result_is_sparse,
            n_symbols=N_SYMBOLS,
            cant_allocate_on_stack=cant_allocate_on_stack,
        ),
        output_path=output_dir / f"matrix_multiplication_benchmark_{matrix_name}.cc",
    )


def generate(output_dir: Path) -> None:
    np.random.seed(42)

    for i, (matrix_name, _filename, matrix) in enumerate(get_matrices()):
        logger.debug(f"Generating matrix {matrix_name}")
        generate_matrix(output_dir, matrix_name, matrix, symforce_result_is_sparse=i > 2, i=i)
