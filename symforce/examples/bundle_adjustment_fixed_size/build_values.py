# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
This file builds a Values with all symbols needed for the fixed-size example.
"""
import symforce.symbolic as sf
from symforce.values import Values


def define_view(index: int) -> Values:
    """
    Creates a symbolic pose + calibration representing a single image
    """
    values = Values()
    values["calibration"] = sf.M(sf.LinearCameraCal.symbolic(f"cal{index}").to_storage())
    values["pose"] = sf.Pose3.symbolic(f"pose{index}")
    return values


def define_feature_match(index: int, match_num: int) -> Values:
    """
    Create a symbolic correspondence definition with the given specs. This includes
    every symbol specific to defining and optimizing a single 2D-2D match.

    Args:
        index (int): Camera index
        match_num (int): Feature match number for this specific target camera
        create_landmark (bool): Use landmark variable parameterization
    """
    values = Values()

    # Source pixel coordinate (camera 0)
    values["source_coords"] = sf.V2.symbolic(f"source_coords_{index}{match_num}")

    # Target pixel coordinate (specified camera index)
    values["target_coords"] = sf.V2.symbolic(f"target_coords_{index}{match_num}")

    # Weight of match
    values["weight"] = sf.Symbol(f"weights_{index}{match_num}")

    values["inverse_range_prior"] = sf.Symbol(f"inverse_range_priors_{index}{match_num}")

    values["inverse_range_prior_sigma"] = sf.Symbol(
        f"inverse_range_prior_sigmas_{index}{match_num}"
    )

    return values


def define_pose_prior(source_cam_index: int, target_cam_index: int) -> Values:
    """
    Create symbols for a pose prior and uncertainty.
    """
    values = Values()

    values["target_T_src"] = sf.Pose3.symbolic(
        f"target_T_src_{source_cam_index}_{target_cam_index}"
    )

    # Square root information matrix of pose estimate [rad, rad, rad, m, m, m]
    values["sqrt_info"] = sf.M66.symbolic(f"pose_sqrt_info_{source_cam_index}_{target_cam_index}")

    return values


def define_objective_costs() -> Values:
    """
    Define parameters for objectives
    """
    values = Values()

    # Robust cost function transition point for reprojection error [px]
    values["reprojection_error_gnc_scale"] = sf.Symbol("reprojection_error_gnc_scale")

    # Robust cost function mu convexity parameter
    values["reprojection_error_gnc_mu"] = sf.Symbol("reprojection_error_gnc_mu")

    return values


def build_values(num_views: int, num_landmarks: int) -> Values:
    """
    Create a Values object with all symbols needed for optimization of a set
    of camera views with sparse feature matches.

    Args:
        num_views (int): Number of camera views
        num_landmarks (int): Fixed number of landmarks

    Returns:
        (Values): Values to optimize + constants
    """
    values = Values()

    values["views"] = []
    values["priors"] = []
    values["matches"] = []

    # Define camera views
    for src_cam_inx in range(num_views):
        values["views"].append(define_view(src_cam_inx))

        priors = []
        for target_cam_inx in range(num_views):
            priors.append(define_pose_prior(src_cam_inx, target_cam_inx))
        values["priors"].append(priors)

    # Define landmarks
    values["landmarks"] = []
    for i in range(num_landmarks):
        values["landmarks"].append(sf.Symbol(f"source_inverse_ranges{i}"))

    # Define correspondences variables from camera 0 to all others
    for v_i in range(num_views - 1):
        matches = []
        for l_i in range(num_landmarks):
            matches.append(define_feature_match(index=v_i, match_num=l_i))
        values["matches"].append(matches)

    values["costs"] = define_objective_costs()

    values["epsilon"] = sf.Symbol("epsilon")

    return values
