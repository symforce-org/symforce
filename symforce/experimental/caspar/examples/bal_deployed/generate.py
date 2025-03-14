# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import symforce

symforce.set_epsilon_to_number(float(10 * np.finfo(np.float32).eps))

import symforce.symbolic as sf
from symforce import typing as T
from symforce.experimental.caspar import CasparLibrary
from symforce.experimental.caspar import memory as mem


@dataclass
class Cam:
    pose: sf.Pose3
    calib: sf.V3


class Point(sf.V3):
    pass


class Pixel(sf.V2):
    pass


caslib = CasparLibrary()


@caslib.add_factor
def fac_reprojection(
    cam: T.Annotated[Cam, mem.Tunable],
    point: T.Annotated[Point, mem.Tunable],
    pixel: T.Annotated[Pixel, mem.Constant],
) -> sf.V2:
    cam_T_world = cam.pose
    intrinsics = cam.calib
    focal_length, k1, k2 = intrinsics
    point_cam = cam_T_world * point
    d = point_cam[2]
    p = -sf.V2(point_cam[:2]) / (d + sf.epsilon() * sf.sign(d))
    r = 1 + k1 * p.squared_norm() + k2 * p.squared_norm() ** 2
    pixel_projected = focal_length * r * p
    err = pixel_projected - pixel
    return err


if __name__ == "__main__":
    caslib.generate(Path(__file__).resolve().parent / "generated")
