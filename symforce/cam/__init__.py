# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Package for symbolic camera models.
"""

from .atan_camera_cal import ATANCameraCal
from .camera import Camera
from .camera_cal import CameraCal
from .double_sphere_camera_cal import DoubleSphereCameraCal
from .equirectangular_camera_cal import EquirectangularCameraCal
from .linear_camera_cal import LinearCameraCal
from .orthographic_camera_cal import OrthographicCameraCal
from .polynomial_camera_cal import PolynomialCameraCal
from .posed_camera import PosedCamera
from .spherical_camera_cal import SphericalCameraCal

# Default generated cam types
# This order came from having originally sorted the subclasses of CameraCal by name, then later
# adding OrthographicCameraCal.
# This order should be maintained, and any new types added at the end to avoid renumbering
# the camera enum types in "symforce/lcmtypes/symforce_types.lcm"
CAM_TYPES = (
    ATANCameraCal,
    DoubleSphereCameraCal,
    EquirectangularCameraCal,
    LinearCameraCal,
    PolynomialCameraCal,
    SphericalCameraCal,
    OrthographicCameraCal,
)
