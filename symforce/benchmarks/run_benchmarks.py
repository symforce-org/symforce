# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
"""
Helper script to run all of the benchmarks, and put timing results into a directory

See README files in each directory for a description of each benchmark
"""

import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path

import argh

from symforce import logger
from symforce import path_util
from symforce import typing as T
from symforce.benchmarks.matrix_multiplication.generate_matrix_multiplication_benchmark import (
    get_matrices,
)

CONFIG = {
    "inverse_compose_jacobian": {
        "double": {
            "gtsam_chained",
            "gtsam_flattened",
            "sophus_chained - double",
            "sym_chained - double",
            "sym_flattened - double",
        },
        "float": {
            "sophus_chained - float",
            "sym_chained - float",
            "sym_flattened - float",
        },
    },
    "robot_3d_localization": {
        "double": {
            "ceres_linearize",
            "ceres_iterate",
            "gtsam_linearize",
            "gtsam_iterate",
            "sym_dynamic_linearize - double",
            "sym_dynamic_iterate - double",
            "sym_fixed_linearize - double",
            "sym_fixed_iterate - double",
        },
        "float": {
            "sym_dynamic_linearize - float",
            "sym_fixed_linearize - float",
            "sym_dynamic_iterate - float",
            "sym_fixed_iterate - float",
        },
    },
}


def run(
    benchmark: str,
    exe_name: str,
    test_name: str,
    out_path: Path,
    stats: T.List[str],
    allow_no_matches: bool = False,
) -> T.Optional[str]:
    # This is wrong if the user changes the build directory
    exe_path = path_util.binary_output_dir() / "bin" / "benchmarks" / exe_name

    # The sparse matrix mult is so slow, run it only once (actually 1M times)
    repeat = 1 if "sparse" in test_name else 10

    # Pin to core 2
    cpu_core = 2

    # Tests are expected to wait 100 ms before doing the good stuff
    wait_time_ms = 90

    cmd = (
        [
            # Pin to a core
            "taskset",
            "-c",
            str(cpu_core),
            # Collect performance stats
            "perf",
            "stat",
            # Repeat and average
            "-r",
            str(repeat),
            # Wait before timing
            f"-D{wait_time_ms}",
        ]
        + stats
        + [
            # Path to binary
            str(exe_path),
            # Name of the test case
            f'"{test_name}"',
        ]
    )

    print(" ".join(cmd))

    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)

    if "No test cases matched" in output:
        msg = f"No test cases for command:\n{' '.join(cmd)}"

        if allow_no_matches:
            logger.warning(msg)
            return None
        else:
            raise ValueError(msg)

    benchmark_dir = out_path / benchmark
    if not benchmark_dir.is_dir():
        benchmark_dir.mkdir()

    file = benchmark_dir / (test_name.replace(" - ", "_") + ".txt")
    file.write_text(output)

    return output


def run_benchmark(
    benchmark: str, benchmark_config: T.Mapping[str, T.Iterable[str]], out_path: Path
) -> None:
    for _, scalar_config in benchmark_config.items():
        for test_name in scalar_config:
            exe_name = f"{benchmark}_benchmark"
            run(benchmark, exe_name, test_name, out_path, ["-d"])


@dataclass(frozen=True)
class MatmulBenchmarkConfig:
    matrix_name: str
    scalar_type: str
    test: str
    M: int
    N: int
    size: int
    nnz: int


def run_matmul_benchmark(
    out_path: Path,
) -> T.Dict[MatmulBenchmarkConfig, T.Optional[T.List[float]]]:
    matrices = get_matrices()

    tests = ["sparse", "flattened", "dense_dynamic", "dense_fixed"]

    results = {}
    for matrix_name, _filename, M in matrices:
        for scalar in ["double", "float"]:
            for test in tests:
                output = run(
                    "matrix_multiplication",
                    f"matrix_multiplication_benchmark_{matrix_name}",
                    f"{test}_{matrix_name} - {scalar}",
                    out_path,
                    ["-x", ",", "-etask-clock,instructions,L1-dcache-loads"],
                    allow_no_matches=True,
                )

                if output is None:
                    matrix_results = None
                else:
                    # Parse n_runs_multiplier out of the log, and divide results by number of runs
                    # to give all metrics per-run
                    gain = float(output.splitlines()[1].split()[-1][:-1])
                    scale = (10 ** 2) * (gain ** 2)

                    matrix_results = [float(l.split(",")[0]) for l in output.splitlines()[-3:]]
                    matrix_results = [x / scale for x in matrix_results]

                results[
                    MatmulBenchmarkConfig(
                        matrix_name,
                        scalar,
                        test,
                        M.shape[0],
                        M.shape[1],
                        M.shape[0] * M.shape[1],
                        M.nnz,
                    )
                ] = matrix_results

    with (out_path / "matrix_multiplication_benchmark_results.pkl").open("wb") as f:
        pickle.dump(results, f)

    print(results)

    return results


@argh.arg(
    "--benchmark",
    help="The name of a particular benchmark to run, instead of running all benchmarks",
)
@argh.arg(
    "--out_dir", help="Directory in which to put results (will be created if it does not exist)"
)
def main(benchmark: str = None, out_dir: str = "benchmark_outputs") -> None:
    out_path = Path(out_dir)
    if not out_path.is_dir():
        out_path.mkdir()

    if benchmark is not None:
        if benchmark == "matrix_multiplication":
            run_matmul_benchmark(out_path)
        else:
            run_benchmark(benchmark, CONFIG[benchmark], out_path)
    else:
        for (  # pylint: disable=redefined-argument-from-local
            benchmark,
            benchmark_config,
        ) in CONFIG.items():
            run_benchmark(benchmark, benchmark_config, out_path)

        run_matmul_benchmark(out_path)


if __name__ == "__main__":
    main.__doc__ = __doc__
    argh.dispatch_command(main)
