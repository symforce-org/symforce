# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from pathlib import Path

import numpy as np

import symforce

symforce.set_epsilon_to_number(float(10 * np.finfo(np.float32).eps))
import torch

import symforce.symbolic as sf
from symforce import typing as T
from symforce.experimental.caspar import CasparLibrary
from symforce.experimental.caspar import memory as mem

"""
We need unique classes to distinguish between values and expectations
"""


class Landmark(sf.V2): ...


class Odometry(sf.V3): ...


class Lidar(sf.V2): ...


class Gnss(sf.V2): ...


caslib = CasparLibrary()


@caslib.add_kernel
def make_poses(
    angle: T.Annotated[sf.Symbol, mem.ReadSequential],
) -> T.Annotated[sf.Pose2, mem.WriteSequential]:
    x = sf.cos(angle) * 10
    y = sf.sin(angle) * 10

    return sf.Pose2.from_tangent([angle, x, y])


@caslib.add_kernel
def get_odometry(
    pose_k: T.Annotated[sf.Pose2, mem.ReadIndexed],
    pose_kp1: T.Annotated[sf.Pose2, mem.ReadIndexed],
) -> T.Annotated[Odometry, mem.WriteSequential]:
    return Odometry(pose_k.local_coordinates(pose_kp1))


@caslib.add_kernel
def get_lidar(
    pose: T.Annotated[sf.Pose2, mem.ReadShared],
    landmark: T.Annotated[Landmark, mem.ReadShared],
) -> T.Annotated[Lidar, mem.WriteSequential]:
    landmark_body = pose.inverse() * landmark
    return Lidar(landmark_body)


@caslib.add_kernel
def get_gnss(
    pose: T.Annotated[sf.Pose2, mem.ReadIndexed],
) -> T.Annotated[Gnss, mem.WriteSequential]:
    return Gnss(pose.t)


@caslib.add_factor
def fac_lidar(
    pose: T.Annotated[sf.Pose2, mem.Tunable],
    landmark: T.Annotated[Landmark, mem.Tunable],
    observed_lidar: T.Annotated[Lidar, mem.Constant],
) -> sf.V2:
    return pose.inverse() * landmark - observed_lidar


@caslib.add_factor
def fac_position(
    pose: T.Annotated[sf.Pose2, mem.Tunable],
    observed_gnss: T.Annotated[Lidar, mem.Constant],
) -> sf.V2:
    return pose.t - observed_gnss


@caslib.add_factor
def fac_odometry(
    pose_k: T.Annotated[sf.Pose2, mem.Tunable],
    pose_kp1: T.Annotated[sf.Pose2, mem.Tunable],
    observed_odom: T.Annotated[Odometry, mem.Constant],
) -> sf.V3:
    return sf.V3(pose_k.local_coordinates(pose_kp1)) - observed_odom


out_dir = Path(__file__).resolve().parent / "generated"
caslib.generate(out_dir)
caslib.compile(out_dir)


# Can also be imported using: lib = caslib.import_lib(out_dir)
from generated import caspar_lib as lib  # type: ignore[import-not-found]

torch.set_default_device("cuda")
N_POSE = 1024
N_LANDMARK = 300
INTERVAL_GNSS = 1
MATCH_PER_LIDAR = 8

arange_pose = torch.arange(N_POSE, dtype=torch.int32)

angles = torch.linspace(0, 2 * np.pi, N_POSE, device="cuda")[None, :]
pose_caspar = torch.empty(mem.caspar_size(sf.Pose2), N_POSE)
lib.make_poses(angles, pose_caspar, N_POSE)

landmarks_caspar = (torch.rand(mem.caspar_size(Landmark), N_LANDMARK) - 0.5) * 10

odometry_caspar = torch.empty(mem.caspar_size(Odometry), N_POSE - 1)
odom_k_indices = arange_pose[:-1]
odom_kp1_indices = arange_pose[1:]
lib.get_odometry(
    pose_caspar, odom_k_indices, pose_caspar, odom_kp1_indices, odometry_caspar, N_POSE - 1
)

gnss_caspar = torch.empty(mem.caspar_size(Gnss), N_POSE // INTERVAL_GNSS)
gnss_indices = torch.arange(0, N_POSE, INTERVAL_GNSS, dtype=torch.int32)
lib.get_gnss(pose_caspar, gnss_indices, gnss_caspar, N_POSE // INTERVAL_GNSS)


lidar_caspar = torch.empty(mem.caspar_size(Lidar), N_POSE * MATCH_PER_LIDAR)
lidar_pose_indices = arange_pose.repeat_interleave(MATCH_PER_LIDAR)
lidar_pose_indices_shared = torch.empty(N_POSE * MATCH_PER_LIDAR, 2, dtype=torch.int32)
lib.shared_indices(lidar_pose_indices, lidar_pose_indices_shared)

_indices_tmp = torch.arange(MATCH_PER_LIDAR)[None:] + torch.arange(N_POSE)[:, None]
lidar_landmark_indices = _indices_tmp.to(torch.int32).ravel() % N_LANDMARK
lidar_landmark_indices_shared = torch.empty(N_POSE * MATCH_PER_LIDAR, 2, dtype=torch.int32)
lib.shared_indices(lidar_landmark_indices, lidar_landmark_indices_shared)
lib.get_lidar(
    pose_caspar,
    lidar_pose_indices_shared,
    landmarks_caspar,
    lidar_landmark_indices_shared,
    lidar_caspar,
    N_POSE * MATCH_PER_LIDAR,
)

pose_stacked = torch.empty(N_POSE, mem.stacked_size(sf.Pose2))
landmarks_stacked = torch.empty(N_LANDMARK, mem.stacked_size(Landmark))
odometry_stacked = torch.empty(N_POSE - 1, mem.stacked_size(Odometry))
gnss_stacked = torch.empty(N_POSE // INTERVAL_GNSS, mem.stacked_size(Gnss))
lidar_stacked = torch.empty(N_POSE * MATCH_PER_LIDAR, mem.stacked_size(Lidar))
lib.Pose2_caspar_to_stacked(pose_caspar, pose_stacked)
lib.Landmark_caspar_to_stacked(landmarks_caspar, landmarks_stacked)
lib.Odometry_caspar_to_stacked(odometry_caspar, odometry_stacked)
lib.Gnss_caspar_to_stacked(gnss_caspar, gnss_stacked)
lib.Lidar_caspar_to_stacked(lidar_caspar, lidar_stacked)


params = lib.SolverParams()
params.diag_init = 1.0
params.solver_iter_max = 200
params.pcg_iter_max = 50
params.pcg_rel_error_exit = 1e-6

solver = lib.GraphSolver(
    params,
    Pose2_num_max=N_POSE,
    Landmark_num_max=N_LANDMARK,
    fac_lidar_num_max=N_POSE * MATCH_PER_LIDAR,
    fac_position_num_max=N_POSE // INTERVAL_GNSS,
    fac_odometry_num_max=N_POSE - 1,
)

pose_stacked_noisy = pose_stacked.clone()
pose_stacked_noisy[:, 2:4] += torch.randn_like(pose_stacked_noisy[:, 2:4])
landmarks_stacked_noisy = landmarks_stacked + torch.randn_like(landmarks_stacked)
solver.set_Pose2_nodes_from_stacked_device(pose_stacked_noisy)
solver.set_Landmark_nodes_from_stacked_device(landmarks_stacked_noisy)

solver.set_fac_lidar_observed_lidar_data_from_stacked_device(lidar_stacked)
solver.set_fac_lidar_pose_indices_from_device(lidar_pose_indices)
solver.set_fac_lidar_landmark_indices_from_device(lidar_landmark_indices)

solver.set_fac_position_observed_gnss_data_from_stacked_device(gnss_stacked)
solver.set_fac_position_pose_indices_from_device(gnss_indices)

solver.set_fac_odometry_observed_odom_data_from_stacked_device(odometry_stacked)
solver.set_fac_odometry_pose_k_indices_from_device(odom_k_indices)
solver.set_fac_odometry_pose_kp1_indices_from_device(odom_kp1_indices)

solver.finish_indices()
solver.solve(print_progress=True)
