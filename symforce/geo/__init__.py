# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Package for symbolic geometric types using the Sympy API, focused on 2D and 3D transforms
for use in robotics and optimization.

Implements strict notions of Storage types, Group types, and LieGroup types.

The design of these concepts is similar to those in GTSAM and Sophus:

    * https://gtsam.org/
    * https://github.com/strasdat/Sophus
"""

from .complex import Complex
from .dual_quaternion import DualQuaternion
from .matrix import *  # pylint: disable=wildcard-import
from .pose2 import Pose2
from .pose2_se2 import Pose2_SE2
from .pose3 import Pose3
from .pose3_se3 import Pose3_SE3
from .quaternion import Quaternion
from .rot2 import Rot2
from .rot3 import Rot3
