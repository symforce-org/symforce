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


@dataclass
class ResidualBlockWithCustomJacobian(ResidualBlock):
    """
    A residual block, with a custom jacobian for the residual

    This should generally only be used if you want to override the jacobian computed by SymForce,
    e.g. to stop derivatives with respect to certain variables or directions, or because the
    jacobian can be analytically simplified in a way that SymForce won't do automatically.  The
    custom_jacobians field should then be filled out with a mapping from all inputs to the residual
    which may be differentiated with respect to, to the desired jacobian of the residual with
    respect to each of those inputs.
    """

    custom_jacobians: T.Dict[T.Element, geo.Matrix]
