# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
A small example of how a kernel can be generated from an annotated symbolic function, and how to
call it.
"""

from pathlib import Path

import numpy as np

import symforce

symforce.set_epsilon_to_number(float(10 * np.finfo(np.float32).eps))

import torch

import symforce.symbolic as sf
from symforce import typing as T
from symforce.experimental.caspar import CasparLibrary
from symforce.experimental.caspar import memory as mem

caslib = CasparLibrary()


@caslib.add_kernel
def example_kernel(
    arg0: T.Annotated[sf.V3, mem.ReadShared],
    arg1: T.Annotated[sf.V6, mem.ReadUnique],
) -> T.Tuple[
    T.Annotated[sf.V2, mem.AddSharedSum],
    T.Annotated[sf.Symbol, mem.WriteIndexed],
]:
    sincos = 2 * sf.V2(sf.sin(arg0[0]), sf.cos(arg0[0]))
    product = arg0[2] * sum(arg1) + 1
    return sincos, product


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "generated"
    caslib.generate(out_dir)
    caslib.compile(out_dir)

    # Can also be imported using: lib = caslib.import_lib(out_dir)
    from symforce.experimental.caspar.examples.kernel_example.generated import (  # type: ignore[import-not-found]
        caspar_lib as lib,
    )

    N = 100
    arg0_stacked = torch.rand(N, sf.V3.storage_dim(), device="cuda")
    arg0_caspar = torch.empty(mem.caspar_size(sf.V3.storage_dim()), N, device="cuda")
    lib.Matrix31_stacked_to_caspar(arg0_stacked, arg0_caspar)

    arg0_indices = torch.randint(0, N, (N,), device="cuda", dtype=torch.int32)
    arg0_indices_shared = torch.empty(N, 2, device="cuda", dtype=torch.int32)
    lib.shared_indices(arg0_indices, arg0_indices_shared)

    arg1_stacked = torch.rand(1, 6, device="cuda")
    arg1_caspar = torch.empty(mem.caspar_size(6), 1, device="cuda")
    lib.Matrix61_stacked_to_caspar(arg1_stacked, arg1_caspar)

    BLOCK_SIZE = 1024
    OUT0_IDX_MAX = 10

    out0_caspar = torch.zeros(mem.caspar_size(2), OUT0_IDX_MAX, device="cuda")
    out0_indices = torch.randint(0, OUT0_IDX_MAX, (N,), device="cuda", dtype=torch.int32)
    out0_indices_shared = torch.empty(N, 2, device="cuda", dtype=torch.int32)
    lib.shared_indices(out0_indices, out0_indices_shared)

    out1_caspar = torch.zeros(mem.caspar_size(1), N, device="cuda")
    out1_indices = torch.randperm(N, device="cuda", dtype=torch.int32)

    lib.example_kernel(
        arg0_caspar,
        arg0_indices_shared,
        arg1_caspar,
        out0_caspar,
        out0_indices_shared,
        out1_caspar,
        out1_indices,
        N,
    )

    out0_sharedsum = torch.zeros(OUT0_IDX_MAX, 2, device="cuda")
    out1_indexed = torch.empty(N, 1, device="cuda")

    lib.Matrix21_caspar_to_stacked(out0_caspar, out0_sharedsum)
    lib.Symbol_caspar_to_stacked(out1_caspar, out1_indexed)

    # Check the results
    sincos = 2 * torch.stack([torch.sin(arg0_stacked[:, 0]), torch.cos(arg0_stacked[:, 0])], dim=1)
    sincos_check = torch.zeros_like(out0_sharedsum)
    for i in range(N):
        sincos_check[out0_indices[i]] += sincos[arg0_indices[i]]
    assert torch.allclose(out0_sharedsum, sincos_check)

    prod_check = torch.zeros(N, 1, device="cuda")
    prod_check[out1_indices] = arg0_stacked[arg0_indices, 2:3] * arg1_stacked.sum() + 1
    assert torch.allclose(out1_indexed, prod_check)

    print("Success!")
