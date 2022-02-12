# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from dataclasses import dataclass

from symforce import geo
from symforce import typing as T


@dataclass
class ResidualBlock:
    """
    A single residual vector, with associated extra values.  The extra values are not used in the
    optimization, but are intended to be additional outputs used for debugging or other purposes.
    """

    residual: geo.Matrix
    extra_values: T.Dataclass
