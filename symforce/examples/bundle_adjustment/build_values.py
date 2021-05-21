"""
This file builds a Values with all symbols needed for the example.
"""
from symforce import cam
from symforce import geo
from symforce.values import Values
from symforce import sympy as sm


def define_view(index: int) -> Values:
    """
    Creates a symbolic pose + calibration representing a single image
    """
    values = Values()
    values["calibration"] = geo.M(cam.LinearCameraCal.symbolic(f"cal{index}").to_storage())
    values["pose"] = geo.Pose3.symbolic(f"pose{index}")
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
    values["source_coords"] = geo.V2.symbolic(f"source_coords_{index}{match_num}")

    # Target pixel coordinate (specified camera index)
    values["target_coords"] = geo.V2.symbolic(f"target_coords_{index}{match_num}")

    # Weight of match
    values["weight"] = sm.Symbol(f"weights_{index}{match_num}")

    values["inverse_range_prior"] = sm.Symbol(f"inverse_range_priors_{index}{match_num}")

    values["inverse_range_prior_sigma"] = sm.Symbol(
        f"inverse_range_prior_sigmas_{index}{match_num}"
    )

    return values


def define_pose_prior(source_cam_index: int, target_cam_index: int) -> Values:
    """
    Create symbols for a pose prior and diagonal uncertainty.
    """
    values = Values()

    values["target_R_src"] = geo.Rot3.symbolic(
        f"target_R_src_{source_cam_index}_{target_cam_index}"
    )

    values["target_t_src"] = geo.V3.symbolic(f"target_t_src_{source_cam_index}_{target_cam_index}")

    # Relative pose constraint weight (scale factor) from 0 to 1
    values["weight"] = sm.Symbol(f"pose_prior_weight_{source_cam_index}_{target_cam_index}")

    # Standard deviation of pose estimate [rad, rad, rad, m, m, m]
    values["sigmas"] = geo.V6.symbolic(f"pose_sigmas_{source_cam_index}_{target_cam_index}")

    return values


def define_objective_costs() -> Values:
    """
    Define parameters for objectives
    """
    values = Values()

    # Robust cost function transition point for reprojection error [px]
    values["reprojection_error_gnc_scale"] = sm.Symbol("reprojection_error_gnc_scale")

    # Robust cost function mu convexity parameter
    values["reprojection_error_gnc_mu"] = sm.Symbol("reprojection_error_gnc_mu")

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
        values["landmarks"].append(sm.Symbol(f"source_inverse_ranges{i}"))

    # Define correspondences variables from camera 0 to all others
    for v_i in range(num_views - 1):
        matches = []
        for l_i in range(num_landmarks):
            matches.append(define_feature_match(index=v_i, match_num=l_i))
        values["matches"].append(matches)

    values["epsilon"] = sm.Symbol("epsilon")

    values["costs"] = define_objective_costs()

    return values
