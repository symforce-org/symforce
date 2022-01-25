from pathlib import Path
import subprocess

from symforce.benchmarks.matrix_multiplication.generate_matrix_multiplication_benchmark import (
    get_matrices,
)
from symforce import typing as T


def run(scalar_type: str, test_name: str, out_path: Path, matrix_name: str) -> T.List[float]:
    exe_path = (
        Path(__file__).resolve().parent.parent.parent
        / "build"
        / "bin"
        / "benchmarks"
        / f"matrix_multiplication_benchmark_{matrix_name}"
    )

    # The sparse matrix mult is so slow, run it only once (actually 1M times)
    repeat = 1 if "sparse" in test_name else 10

    # Pin to core 2
    cpu_core = 2

    # Tests are expected to wait 100 ms before doing the good stuff
    wait_time_ms = 90

    cmd = [
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
        "-x",
        ",",
        "-etask-clock,instructions,L1-dcache-loads",
        # Wait before timing
        f"-D{wait_time_ms}",
        # Path to binary
        str(exe_path),
        # Name of the test case
        f'"{test_name}"',
    ]

    print(" ".join(cmd))

    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)

    benchmark_dir = out_path / "matrix_multiplication"
    if not benchmark_dir.is_dir():
        benchmark_dir.mkdir()

    if "No test cases matched" in output:
        raise ValueError(f"No test cases for command:\n{' '.join(cmd)}")

    results = [float(l.split(",")[0]) for l in output.splitlines()[-3:]]
    return results


def main(out_dir: str = "matrix_benchmark_outputs") -> None:
    """
    Helper script to get profiling data.
    """
    out_path = Path(out_dir)
    if not out_path.is_dir():
        out_path.mkdir()

    matrices = get_matrices()

    tests = ["sparse", "flattened", "dense_dynamic", "dense_fixed"]

    results = {}
    for matrix_name, _filename, M in matrices[:5]:
        for scalar in ["double", "float"]:
            for test in tests:
                results[
                    (
                        matrix_name,
                        scalar,
                        test,
                        M.shape[0],
                        M.shape[1],
                        M.shape[0] * M.shape[1],
                        len(M.nonzero()[0]),
                    )
                ] = run(scalar, f"{matrix_name}_{test} - {scalar}", out_path, matrix_name)

    print(results)


if __name__ == "__main__":
    import argh

    argh.dispatch_command(main)
