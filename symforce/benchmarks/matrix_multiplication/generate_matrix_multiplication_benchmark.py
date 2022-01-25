import dataclasses
import numpy as np
import os
from pathlib import Path
import re
import scipy.io

from symforce import codegen
from symforce import geo
from symforce import python_util
from symforce import sympy as sm
from symforce import typing as T
from symforce.codegen import template_util
from symforce.expressions import unary_binary_expression_gen, op_probabilities
from symforce.values import Values

OPS_PER_ENTRY = 2
N_SYMBOLS = 2
N_MATRICES = 5


def get_matrices() -> T.List:
    matrices = []

    for path in (Path(__file__).parent / "matrices").iterdir():
        if not path.is_dir():
            continue

        matrix_name = path.name
        for filename in filter(lambda p: p.suffix == ".mtx", path.iterdir()):
            matrix = scipy.io.mmread(filename)
            if matrix.shape[0] > 1 and matrix.shape[1] > 1:
                break
        else:
            raise FileNotFoundError(f"Didn't find non-vector Matrix Market file in {path}")
        matrices.append(
            (
                matrix_name.replace(" ", "").replace("-", "_"),
                filename,  # pylint: disable=undefined-loop-variable
                matrix,
            )
        )

    matrices.sort(key=lambda m: m[2].nonzero()[0].size)

    return matrices[:N_MATRICES]


def generate_tests(output_dir: Path) -> None:
    matrices = get_matrices()

    for i, (matrix_name, _, matrix) in enumerate(matrices):
        template_util.render_template(
            template_path=os.fspath(
                Path(__file__).parent / "matrix_multiplication_benchmark.cc.jinja"
            ),
            data=dict(
                matrix_name=matrix_name,
                matrix_name_camel=python_util.snakecase_to_camelcase(matrix_name),
                N=matrix.shape[0],
                M=matrix.shape[1],
                size2=(100.0 if i < 3 else 1.0),
                is_sparse=i > 2,
                n_symbols=N_SYMBOLS,
            ),
            output_path=os.fspath(output_dir / f"matrix_multiplication_benchmark_{matrix_name}.cc"),
            template_dir=os.fspath(Path(__file__).parent),
        )


def _make_return_dynamic(generated_file: Path, matrix: np.ndarray) -> None:
    generated_file.write_text(
        generated_file.read_text()
        .replace(
            f"Eigen::Matrix<Scalar, {matrix.shape[0]}, {matrix.shape[1]}>",
            "Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>",
        )
        .replace("> _result;", f"> _result({matrix.shape[0]}, {matrix.shape[1]});")
    )


def _fix_non_finite_values(generated_file: Path) -> None:
    generated_file.write_text(re.sub("NAN|INFINITY|zoo", "1.0", generated_file.read_text()))


def generate_matrix(
    output_dir: Path,
    matrix_name: str,
    matrix: T.Union[np.ndarray, scipy.sparse.coo_matrix],
    result_is_sparse: bool,
) -> None:
    nnz = matrix.nonzero()[0].size

    if not isinstance(matrix, np.ndarray):
        matrix = matrix.todense()

    symbols = sm.symbols(f"x:{N_SYMBOLS}")
    gen = unary_binary_expression_gen.UnaryBinaryExpressionGen(
        unary_ops=[op_probabilities.OpProbability("neg", lambda x: -x, 3),],
        binary_ops=[
            op_probabilities.OpProbability("add", lambda x, y: x + y, 4),
            op_probabilities.OpProbability("sub", lambda x, y: x - y, 2),
            op_probabilities.OpProbability("mul", lambda x, y: x * y, 5),
            op_probabilities.OpProbability("div", lambda x, y: x / y, 1),
        ],
        leaves=[-2, -1, 1, 2] + list(symbols),
    )

    def compute_A(*symbols: T.List[sm.Symbol]) -> geo.Matrix:
        exprs = gen.build_expr_vec(OPS_PER_ENTRY * nnz, nnz).to_flat_list()
        result = geo.Matrix(*matrix.shape)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] != 0:
                    result[i, j] = exprs.pop()
        return result

    A = compute_A(*symbols)
    B = compute_A(*symbols)
    ATB = A.T * B

    from sympy.simplify import cse_opts

    config = codegen.CppConfig(cse_optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)])
    config_noinline = dataclasses.replace(config, force_no_inline=True)

    data = codegen.Codegen(
        inputs=Values(**{s.name: s for s in symbols}),
        outputs=Values(result=A),
        config=config,
        name=f"{matrix_name}_compute_a",
        sparse_matrices=["result"],
        return_key="result",
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)
    _fix_non_finite_values(Path(data["generated_files"][0]))

    data = codegen.Codegen(
        inputs=Values(**{s.name: s for s in symbols}),
        outputs=Values(result=B),
        config=config,
        name=f"{matrix_name}_compute_b",
        sparse_matrices=["result"],
        return_key="result",
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)
    _fix_non_finite_values(Path(data["generated_files"][0]))

    compute_a_dense = codegen.Codegen(
        inputs=Values(**{s.name: s for s in symbols}),
        outputs=Values(result=A),
        config=config,
        name=f"{matrix_name}_compute_a_dense",
        return_key="result",
    )

    data = compute_a_dense.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    _fix_non_finite_values(Path(data["generated_files"][0]))

    assert compute_a_dense.name is not None
    compute_a_dense.name += "_dynamic"
    data = compute_a_dense.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    _make_return_dynamic(Path(data["generated_files"][0]), matrix)
    _fix_non_finite_values(Path(data["generated_files"][0]))

    compute_b_dense = codegen.Codegen(
        inputs=Values(**{s.name: s for s in symbols}),
        outputs=Values(result=B),
        config=config,
        name=f"{matrix_name}_compute_b_dense",
        return_key="result",
    )

    data = compute_b_dense.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    _fix_non_finite_values(Path(data["generated_files"][0]))

    assert compute_b_dense.name is not None
    compute_b_dense.name += "_dynamic"
    data = compute_b_dense.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    _make_return_dynamic(Path(data["generated_files"][0]), matrix)
    _fix_non_finite_values(Path(data["generated_files"][0]))

    data = codegen.Codegen(
        inputs=Values(**{s.name: s for s in symbols}),
        outputs=Values(result=ATB),
        config=config_noinline,
        name=f"{matrix_name}_compute_at_b",
        return_key="result",
        sparse_matrices=["result"] if result_is_sparse else [],
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)
    _fix_non_finite_values(Path(data["generated_files"][0]))

    cant_allocate_on_stack = (
        matrix.shape[0] * matrix.shape[1] * 8 > 131072
        or matrix.shape[0] * matrix.shape[0] * 8 > 131072
    )

    if cant_allocate_on_stack and not result_is_sparse:
        _make_return_dynamic(Path(data["generated_files"][0]), matrix)


def generate_matrices(output_dir: Path) -> None:
    np.random.seed(42)

    for i, (matrix_name, _filename, matrix) in enumerate(get_matrices()):
        print(f"Generating matrix {matrix_name}")
        generate_matrix(output_dir, matrix_name, matrix, result_is_sparse=i > 2)
