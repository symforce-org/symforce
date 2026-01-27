# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
import symforce

USE_DOUBLE = False  # Set USE_DOUBLE to True to have Caspar use double precision.
symforce.set_epsilon_to_number(1e-15 if USE_DOUBLE else 1e-6)
torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64 if USE_DOUBLE else torch.float32)


import symforce.symbolic as sf
from symforce import typing as T
from symforce.experimental.caspar import CasparLibrary
from symforce.experimental.caspar import memory as mem

caslib = CasparLibrary(dtype=mem.DType.DOUBLE if USE_DOUBLE else mem.DType.FLOAT)


# Define the types used in the factor graph.
# This is not required, but is recommended to have Caspar generate descriptive function names.
class Pose(sf.Pose3): ...


class Landmark(sf.V3): ...


class posMeasurement(sf.V3): ...


class posSensorOffset(sf.V3): ...


class OdometryMeasurement(sf.V6): ...


class LandmarkMeasurement(sf.V3): ...


class LandmarkSensorOffset(sf.Pose3): ...


# Define the kernels we use to generate synthetic data.
@caslib.add_kernel
def make_poses(
    angle: T.Annotated[sf.Symbol, mem.ReadSequential],
) -> T.Annotated[sf.Pose3, mem.WriteSequential]:
    x = sf.sin(angle) * 10
    y = -sf.cos(angle) * 10
    z = 0
    return sf.Pose3(sf.Rot3.from_tangent([0, 0, angle]), sf.V3(x, y, z))


@caslib.add_kernel
def get_pos_measurements(
    pose: T.Annotated[sf.Pose3, mem.ReadIndexed],
    pos_sensor_offset: T.Annotated[posSensorOffset, mem.ReadUnique],
) -> T.Annotated[posMeasurement, mem.WriteSequential]:
    return posMeasurement(pose * pos_sensor_offset)


@caslib.add_kernel
def get_odometry_measurements(
    poses: T.Annotated[mem.Pair[sf.Pose3], mem.ReadPair],
) -> T.Annotated[OdometryMeasurement, mem.WriteSequential]:
    return OdometryMeasurement(poses[0].local_coordinates(poses[1]))


@caslib.add_kernel
def get_landmark_measurements(
    pose: T.Annotated[sf.Pose3, mem.ReadShared],
    landmark_sensor_extrinsics: T.Annotated[LandmarkSensorOffset, mem.ReadUnique],
    landmark: T.Annotated[Landmark, mem.ReadShared],
) -> T.Annotated[LandmarkMeasurement, mem.WriteSequential]:
    landmark_body = pose.compose(landmark_sensor_extrinsics).inverse() * landmark
    return LandmarkMeasurement(landmark_body)


# Define the factors/loss functions we use in the factor graph.
@caslib.add_factor
def pos_error(
    pose: T.Annotated[Pose, mem.TunableShared],
    pos_sensor_offset: T.Annotated[posSensorOffset, mem.TunableUnique],
    pos_meas: T.Annotated[posMeasurement, mem.ConstantSequential],
) -> sf.V3:
    return get_pos_measurements(pose, pos_sensor_offset) - pos_meas


@caslib.add_factor
def odometry_error(
    poses: T.Annotated[mem.Pair[Pose], mem.TunablePair],
    odom_meas: T.Annotated[OdometryMeasurement, mem.ConstantSequential],
) -> sf.V6:
    return get_odometry_measurements(poses) - odom_meas


@caslib.add_factor
def landmark_error(
    pose: T.Annotated[Pose, mem.TunableShared],
    landmark_sensor_extrinsics: T.Annotated[LandmarkSensorOffset, mem.TunableUnique],
    landmark: T.Annotated[Landmark, mem.TunableShared],
    landmark_meas: T.Annotated[LandmarkMeasurement, mem.ConstantSequential],
) -> sf.V3:
    return get_landmark_measurements(pose, landmark_sensor_extrinsics, landmark) - landmark_meas


def to_tensor(storage: sf.Storage) -> torch.Tensor:
    """Helper function to convert sf.Storage to torch.Tensor"""
    return torch.tensor([[float(a) for a in storage.evalf().to_storage()]])


# Generate, compile and load the Caspar library.
out_dir = Path(__file__).resolve().parent / "generated"
caslib.generate(out_dir)  # Can be commented out after the first run to avoid regenerating (slow)
caslib.compile(out_dir)  # Can be commented out after the first run to avoid recompiling (slow)

# Can also be imported using:
# lib = caslib.import_lib(out_dir)
from generated import caspar_lib as lib  # type: ignore[import-not-found, unused-ignore]

# Generate synthetic data using the kernels we defined above
# Caspar uses it's own memorylayout for efficient acces,
# which is why we have {data}_caspar and {data}_stacked variables.
N_POSE = 8000
N_LANDMARK = 300
INTERVAL_GNSS = 12
MATCH_PER_LIDAR = 3

N_GNSS = 1 + (N_POSE - 1) // INTERVAL_GNSS
N_LANDMARK_ERROR = N_POSE * MATCH_PER_LIDAR
N_LM_HALF = N_LANDMARK_ERROR // 2

pos_sensor_offset = to_tensor(posSensorOffset.from_tangent([0, 0, 0]))
landmark_sensor_offset = to_tensor(LandmarkSensorOffset.from_tangent([0, 0, 0, 0, 0, 0]))

angles = torch.linspace(0, 2 * np.pi, N_POSE)[None, :]
pose_caspar = torch.empty(mem.caspar_size(Pose), N_POSE)
lib.make_poses(angles, pose_caspar, N_POSE)

landmarks_caspar = (torch.rand(mem.caspar_size(Landmark), N_LANDMARK) - 0.5) * 10

odometry_caspar = torch.empty(mem.caspar_size(OdometryMeasurement), N_POSE - 1)
lib.get_odometry_measurements(pose_caspar, odometry_caspar, N_POSE - 1)

pos_meas_caspar = torch.empty(mem.caspar_size(posMeasurement), N_GNSS)
pos_indices = torch.arange(0, N_POSE, INTERVAL_GNSS, dtype=torch.int32)
lib.get_pos_measurements(pose_caspar, pos_indices, pos_sensor_offset, pos_meas_caspar, N_GNSS)

landmark_meas_caspar = torch.empty(mem.caspar_size(LandmarkMeasurement), N_LANDMARK_ERROR)
ladmark_error_pose_indices = torch.arange(N_POSE, dtype=torch.int32).repeat_interleave(
    MATCH_PER_LIDAR
)
ladmark_error_pose_indices_shared = torch.empty(N_LANDMARK_ERROR, 2, dtype=torch.int32)
lib.shared_indices(ladmark_error_pose_indices, ladmark_error_pose_indices_shared)


_indices_tmp = torch.arange(MATCH_PER_LIDAR)[None:] + torch.arange(N_POSE)[:, None]
landmark_error_landmark_indices = _indices_tmp.to(torch.int32).ravel() % N_LANDMARK
landmark_error_landmark_indices_shared = torch.empty(N_LANDMARK_ERROR, 2, dtype=torch.int32)
lib.shared_indices(landmark_error_landmark_indices, landmark_error_landmark_indices_shared)

lib.get_landmark_measurements(
    pose_caspar,
    ladmark_error_pose_indices_shared,
    landmark_sensor_offset,
    landmarks_caspar,
    landmark_error_landmark_indices_shared,
    landmark_meas_caspar,
    N_LANDMARK_ERROR,
)


# Map the generated Caspar data to regular array of structs (AOS) format.
pose_stacked = torch.empty(N_POSE, mem.stacked_size(Pose))
lib.Pose_caspar_to_stacked(pose_caspar, pose_stacked)
landmarks_stacked = torch.empty(N_LANDMARK, mem.stacked_size(Landmark))
lib.Landmark_caspar_to_stacked(landmarks_caspar, landmarks_stacked)
odometry_stacked = torch.empty(N_POSE - 1, mem.stacked_size(OdometryMeasurement))
lib.OdometryMeasurement_caspar_to_stacked(odometry_caspar, odometry_stacked)
pos_meas_stacked = torch.empty(N_GNSS, mem.stacked_size(posMeasurement))
lib.posMeasurement_caspar_to_stacked(pos_meas_caspar, pos_meas_stacked)
landmark_meas_stacked = torch.empty(N_LANDMARK_ERROR, mem.stacked_size(LandmarkMeasurement))
lib.LandmarkMeasurement_caspar_to_stacked(landmark_meas_caspar, landmark_meas_stacked)


# Add some noise to the data.
pose_stacked_noisy = pose_stacked.clone()
pose_stacked_noisy[:, 4:7] += torch.randn_like(pose_stacked_noisy[:, 4:7])
landmarks_stacked_noisy = landmarks_stacked + torch.randn_like(landmarks_stacked)


# Create the solver, and set parameters and configure allocation sizes for the solver.
params = lib.SolverParams()
params.diag_init = 2
params.solver_iter_max = 50
params.pcg_iter_max = 20
params.pcg_rel_error_exit = 1e-2
params.diag_min = 1e-16 if USE_DOUBLE else 1e-6

solver = lib.GraphSolver(
    params,
    Pose_num_max=N_POSE,
    Landmark_num_max=N_LANDMARK,
    posSensorOffset_num_max=1,
    LandmarkSensorOffset_num_max=1,
    pos_error_num_max=N_GNSS,
    landmark_error_num_max=N_LANDMARK_ERROR,
    odometry_error_num_max=N_POSE - 1,
)


solver.set_posSensorOffset_nodes_from_stacked_device(pos_sensor_offset)
solver.set_LandmarkSensorOffset_nodes_from_stacked_device(landmark_sensor_offset)

# To demonstrade how to update the solver dynamically we start by loading and optimizing only half the problem.
landmarks_in_first_halt = int(landmark_error_landmark_indices[:N_LM_HALF].amax().item() + 1)
solver.set_Pose_num(N_POSE // 2)
solver.set_Landmark_num(landmarks_in_first_halt)
solver.set_Pose_nodes_from_stacked_device(pose_stacked_noisy[: N_POSE // 2])
solver.set_Landmark_nodes_from_stacked_device(landmarks_stacked_noisy[:landmarks_in_first_halt])

solver.set_landmark_error_num(N_LM_HALF)
solver.set_landmark_error_pose_indices_from_device(ladmark_error_pose_indices[:N_LM_HALF])
solver.set_landmark_error_landmark_indices_from_device(landmark_error_landmark_indices[:N_LM_HALF])
solver.set_landmark_error_landmark_meas_data_from_stacked_device(landmark_meas_stacked[:N_LM_HALF])


solver.set_pos_error_num(N_GNSS // 2)
solver.set_pos_error_pose_indices_from_device(pos_indices[: N_GNSS // 2])
solver.set_pos_error_pos_meas_data_from_stacked_device(pos_meas_stacked[: N_GNSS // 2])

solver.set_odometry_error_num(N_POSE // 2 - 1)
solver.set_odometry_error_odom_meas_data_from_stacked_device(odometry_stacked[: N_POSE // 2])

solver.solve(print_progress=True)

# After optimizing the first half we upload the remaining data and optimize the full problem.
solver.set_Pose_num(N_POSE)
solver.set_Landmark_num(N_LANDMARK)
solver.set_Pose_nodes_from_stacked_device(pose_stacked_noisy[N_POSE // 2 :], offset=N_POSE // 2)
solver.set_Landmark_nodes_from_stacked_device(
    landmarks_stacked_noisy[landmarks_in_first_halt:], offset=landmarks_in_first_halt
)

solver.set_landmark_error_num(N_LANDMARK_ERROR)
solver.set_landmark_error_pose_indices_from_device(ladmark_error_pose_indices)
solver.set_landmark_error_landmark_indices_from_device(landmark_error_landmark_indices)
solver.set_landmark_error_landmark_meas_data_from_stacked_device(
    landmark_meas_stacked[N_LM_HALF:], offset=N_LM_HALF
)
solver.set_pos_error_num(N_GNSS)
solver.set_pos_error_pose_indices_from_device(pos_indices)
solver.set_pos_error_pos_meas_data_from_stacked_device(
    pos_meas_stacked[N_GNSS // 2 :], offset=N_GNSS // 2
)

solver.set_odometry_error_num(N_POSE - 1)
solver.set_odometry_error_odom_meas_data_from_stacked_device(
    odometry_stacked[N_POSE // 2 - 1 :], offset=N_POSE // 2 - 1
)

solver.solve(print_progress=True)
