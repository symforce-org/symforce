"""
Package for symbolic geometric types using the Sympy API, focused on 2D and 3D transforms
for use in robotics and optimization.

Implements strict notions of Storage types, Group types, and LieGroup types.

The design of these concepts is similar to those in GTSAM and Sophus:

    * https://gtsam.org/
    * https://github.com/strasdat/Sophus
"""

from .base import Group
from .base import LieGroup
from .base import Storage
from .complex import Complex
from .dual_quaternion import DualQuaternion
from .matrix import *  # pylint: disable=wildcard-import
from .quaternion import Quaternion

from .pose2 import Pose2
from .pose3 import Pose3
from .rot2 import Rot2
from .rot3 import Rot3

# Shorthand
M = Matrix
