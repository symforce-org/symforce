# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import importlib
import unittest

import numpy as np

import symforce
from symforce import path_util
from symforce.codegen import codegen_util
from symforce.test_util import TestCase

TEST_DATA_DIR = (
    path_util.symforce_data_root()
    / "test"
    / "symforce_function_codegen_test_data"
    / symforce.get_symbolic_api()
    / "symforce_pytorch_codegen_test"
)


class SymforcePyTorchTest(TestCase):
    """
    Tests calling generated PyTorch functions
    """

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "Requires PyTorch")
    def test_backend_test_function(self) -> None:
        import torch

        backend_test_function = codegen_util.load_generated_function(
            "backend_test_function", TEST_DATA_DIR
        )

        # Try with default scalar cpu tensors
        backend_test_function(torch.tensor(1.0), torch.tensor(2.0))

        # With broadcasting
        x = torch.tensor(1.0).tile((1, 2, 3))
        y = torch.tensor(2.0).tile((5, 1, 3))
        results = backend_test_function(x, y)
        self.assertEqual(results[0].shape, ())  # A constant
        self.assertEqual(results[-1].shape, (5, 2, 3))  # A function of x and y

        # With a custom dtype
        results = backend_test_function(
            torch.tensor(1.0, dtype=torch.float64), torch.tensor(2.0, dtype=torch.float64)
        )
        self.assertEqual(results[0].dtype, torch.float64)
        self.assertEqual(results[-1].dtype, torch.float64)

        # With tensor_kwargs
        results = backend_test_function(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(2.0, dtype=torch.float64),
            tensor_kwargs={"dtype": torch.float64},
        )
        self.assertEqual(results[0].dtype, torch.float64)
        self.assertEqual(results[-1].dtype, torch.float64)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "Requires PyTorch")
    def test_vector_matrix_args(self) -> None:
        import torch

        pytorch_func = codegen_util.load_generated_function("pytorch_func", TEST_DATA_DIR)

        a_out, b_out, c_out, d_out, e_out, f_out = pytorch_func(
            a=torch.tensor(1.0).tile((1, 2)),
            b=torch.tensor([1.0]).tile((1, 2, 1)),
            c=torch.tensor([1.0, 2.0, 3.0]).tile(3, 2, 1),
            d=torch.tensor([[1.0, 2.0], [3.0, 4.0]]).tile(1, 1, 1, 1),
            e=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).tile(1, 1, 1),
            f=torch.tensor(np.eye(6)).tile(1, 1, 1, 1),
        )
        self.assertEqual(a_out.shape, (1, 2))
        self.assertEqual(b_out.shape, (1, 2, 1))
        self.assertEqual(c_out.shape, (3, 2, 3))
        self.assertEqual(d_out.shape, (1, 1, 2, 2))
        self.assertEqual(e_out.shape, (1, 1, 5))
        self.assertEqual(f_out.shape, (1, 1, 6, 6))


if __name__ == "__main__":
    SymforcePyTorchTest.main()
