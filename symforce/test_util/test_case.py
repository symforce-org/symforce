import numpy as np
import os
import random
import sys
import unittest
import logging

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

    def setUp(self):
        # type: () -> None
        np.random.seed(42)
        random.seed(42)
        # Store verbosity flag so tests can use
        self.verbose = ("-v" in sys.argv) or ("--verbose" in sys.argv)

    def assertNear(
        self,
        actual,  # type: T.Any
        desired,  # type: T.Any
        places=7,  # type: int
        msg="",  # type: str
        verbose=True,  # type: bool
    ):
        # type: (...) -> None
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
        actual,  # type: LieGroupOpsType
        desired,  # type: LieGroupOpsType
        places=7,  # type: int
        msg="",  # type: str
        verbose=True,  # type: bool
    ):
        # type: (...) -> None
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
    def compile_and_run_cpp(package_dir, executable_names, make_args=tuple()):
        # type: (str, T.Union[str, T.Sequence[str]], T.Sequence[str]) -> None
        """
        Compile package using makefile in package_dir, then execute the executable with
        name executable_name.
        """

        # Build package
        make_cmd = ["make", "-C", package_dir]
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
