from pathlib import Path
import subprocess

from symforce import typing as T

CONFIG = {
    "inverse_compose_jacobian": {
        "double": {
            "gtsam_chained",
            "gtsam_flattened",
            "sophus_chained - double",
            "sym_chained - double",
            "sym_flattened - double",
        },
        "float": {"sophus_chained - float", "sym_chained - float", "sym_flattened - float",},
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


def run(benchmark: str, scalar_type: str, test_name: str, out_path: Path) -> None:
    # This is wrong if the user changes the build directory
    exe_path = (
        Path(__file__).resolve().parent.parent.parent
        / "build"
        / "bin"
        / "benchmarks"
        / f"{benchmark}_benchmark"
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
        # Detailed stats
        "-d",
        # Repeat and average
        "-r",
        str(repeat),
        # Wait before timing
        f"-D{wait_time_ms}",
        # Path to binary
        str(exe_path),
        # Name of the test case
        f'"{test_name}"',
    ]

    print(" ".join(cmd))

    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)

    benchmark_dir = out_path / benchmark
    if not benchmark_dir.is_dir():
        benchmark_dir.mkdir()

    file = benchmark_dir / (test_name.replace(" - ", "_") + ".txt")
    file.write_bytes(output)


def run_benchmark(
    benchmark: str, benchmark_config: T.Mapping[str, T.Iterable[str]], out_path: Path
) -> None:
    for scalar, scalar_config in benchmark_config.items():
        for test_name in scalar_config:
            run(benchmark, scalar, test_name, out_path)


def main(benchmark: str = None, out_dir: str = "benchmark_outputs") -> None:
    """
    Helper script to get profiling data.
    """
    out_path = Path(out_dir)
    if not out_path.is_dir():
        out_path.mkdir()

    if benchmark is not None:
        run_benchmark(benchmark, CONFIG[benchmark], out_path)
    else:
        for (  # pylint: disable=redefined-argument-from-local
            benchmark,
            benchmark_config,
        ) in CONFIG.items():
            run_benchmark(benchmark, benchmark_config, out_path)


if __name__ == "__main__":
    import argh

    argh.dispatch_command(main)
