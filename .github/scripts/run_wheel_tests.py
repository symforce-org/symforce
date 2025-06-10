import asyncio
import os
import platform
import sys
import textwrap
from pathlib import Path


async def run_test(
    semaphore: asyncio.Semaphore, output_lock: asyncio.Lock, test_file: Path, symbolic_api: str
) -> bool:
    async with semaphore:
        args = [test_file]

        # Codegen is different on macOS; there's some nondeterminism in SymEngine, based on e.g.
        # unordered_map iteration order, that we need to fix.  For now, this is fine, so we check
        # that the tests pass, but allow them to generate different code than is checked in.
        include_update_flag = (
            platform.system() == "Darwin"
            and test_file.name.endswith("_codegen_test.py")
            and symbolic_api == "symengine"
        )
        if include_update_flag:
            args.append("--update")

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            *args,
            env=dict(os.environ, SYMFORCE_WHEEL_TESTS="1", SYMFORCE_SYMBOLIC_API=symbolic_api),
            stderr=asyncio.subprocess.STDOUT,
            stdout=asyncio.subprocess.PIPE,
        )

        stdout, _ = await proc.communicate()

        if proc.returncode != 0:
            async with output_lock:
                print(
                    textwrap.dedent(
                        """
                        --------------------------------------------------------------------------------
                        Test {test_file} on api {symbolic_api} failed with output:
                        {output}
                        --------------------------------------------------------------------------------
                        """
                    ).format(test_file=test_file, symbolic_api=symbolic_api, output=stdout.decode())
                )
            return False

        return True


async def main() -> int:
    project = Path(sys.argv[1])

    tests = project.glob("test/*_test.py")
    semaphore = asyncio.Semaphore(os.cpu_count())
    output_lock = asyncio.Lock()
    tests = [(test, symbolic_api) for test in tests for symbolic_api in ("symengine", "sympy")]
    results = await asyncio.gather(
        *[run_test(semaphore, output_lock, test, symbolic_api) for test, symbolic_api in tests]
    )
    total_tests = len(results)
    succeeded_tests = sum(results)
    print(f"Ran {total_tests}, {succeeded_tests} passed, {total_tests - succeeded_tests} failed")
    if total_tests != succeeded_tests:
        for result, (test, symbolic_api) in zip(results, tests):
            if not result:
                print(f"    {test} ({symbolic_api})")
    if total_tests != succeeded_tests:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
