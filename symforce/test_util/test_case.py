import multiprocessing
import numpy as np
import os
import random
import re
import sys
import unittest
import logging

import symforce
from symforce import geo
from symforce.ops import interfaces
from symforce import logger
from symforce import python_util
from symforce import types as T
from symforce.ops import StorageOps
from symforce.ops import GroupOps
from symforce.ops import LieGroupOps
from symforce.test_util.test_case_mixin import SymforceTestCaseMixin


class TestCase(unittest.TestCase, SymforceTestCaseMixin):
    """
    Base class for symforce tests. Adds some useful helpers.
    """

    LieGroupOpsType = T.Union[interfaces.LieGroup, T.Scalar]

    # Set by the --run_slow_tests flag to indicate that we should run all tests even
    # if we're on the SymPy backend
    _RUN_SLOW_TESTS = False

    @staticmethod
    def should_run_slow_tests() -> bool:

        # NOTE(aaron):  This needs to be accessible before main() is called, so we do it here
        # instead.  This should also be called from main to make sure it runs at least once
        if "--run_slow_tests" in sys.argv:
            TestCase._RUN_SLOW_TESTS = True
            sys.argv.remove("--run_slow_tests")
        return TestCase._RUN_SLOW_TESTS

    @staticmethod
    def main() -> None:
        """
        Call this to run all tests in scope.
        """
        TestCase.should_run_slow_tests()
        SymforceTestCaseMixin.main()

    def setUp(self) -> None:
        np.random.seed(42)
        random.seed(42)
        # Store verbosity flag so tests can use
        self.verbose = ("-v" in sys.argv) or ("--verbose" in sys.argv)

    def assertNear(
        self, actual: T.Any, desired: T.Any, places: int = 7, msg: str = "", verbose: bool = True,
    ) -> None:
        """
        Check that two elements are close. Handles sequences, scalars, and geometry types
        using StorageOps.
        """
        return np.testing.assert_almost_equal(
            actual=np.array(StorageOps.evalf(StorageOps.to_storage(actual)), dtype=np.double),
            desired=np.array(StorageOps.evalf(StorageOps.to_storage(desired)), dtype=np.double),
            decimal=places,
            err_msg=msg,
            verbose=verbose,
        )

    def assertLieGroupNear(
        self,
        actual: LieGroupOpsType,
        desired: LieGroupOpsType,
        places: int = 7,
        msg: str = "",
        verbose: bool = True,
    ) -> None:
        """
        Check that two LieGroup elements are close.
        """
        epsilon = 10 ** (-max(9, places + 1))
        # Compute the tangent space pertubation around `actual` that produces `desired`
        local_coordinates = LieGroupOps.local_coordinates(actual, desired, epsilon=epsilon)
        # Compute the identity tangent space pertubation to compare against
        identity = geo.Matrix.zeros(LieGroupOps.tangent_dim(actual), 1)
        return np.testing.assert_almost_equal(
            actual=StorageOps.evalf(local_coordinates),
            desired=StorageOps.to_storage(identity),
            decimal=places,
            err_msg=msg,
            verbose=verbose,
        )

    @staticmethod
    def compile_and_run_cpp(
        package_dir: str,
        executable_names: T.Union[str, T.Sequence[str]],
        make_args: T.Sequence[str] = tuple(),
    ) -> None:
        """
        Compile package using makefile in package_dir, then execute the executable with
        name executable_name.
        """

        # Build package
        make_cmd = ["make", "-C", package_dir, "-j{}".format(multiprocessing.cpu_count() - 1)]
        if make_args:
            make_cmd += make_args
        if logger.level != logging.DEBUG:
            make_cmd.append("--quiet")
        python_util.execute_subprocess(make_cmd)

        # Run executable(s)
        if isinstance(executable_names, str):
            # We just have one executable
            python_util.execute_subprocess(os.path.join(package_dir, executable_names))
        else:
            # We have a list of executables
            for name in executable_names:
                python_util.execute_subprocess(os.path.join(package_dir, name))


def requires_sympy(func: T.Callable) -> T.Callable:
    """
    Decorator to mark a test to only run on the SymPy backend, and skip otherwise
    """
    backend = symforce.get_backend()
    if backend != "sympy":
        return unittest.skip("This test only runs on the SymPy backend")(func)
    else:
        return func


def slow_on_sympy(func: T.Callable) -> T.Callable:
    """
    Decorator to mark a test as slow on the sympy backend.  Will be skipped unless passed the
    --run_slow_tests flag
    """
    backend = symforce.get_backend()
    if backend == "sympy" and not TestCase.should_run_slow_tests():
        return unittest.skip("This test is too slow on the SymPy backend")(func)
    else:
        return func
