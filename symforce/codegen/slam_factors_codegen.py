import os

from symforce import cam
from symforce import codegen
from symforce import geo
from symforce import sympy as sm
from symforce import types as T
from symforce.opt.noise_models import BarronNoiseModel


def inverse_range_landmark_prior_residual(
    landmark_inverse_range: T.Scalar,
    inverse_range_prior: T.Scalar,
    weight: T.Scalar,
    sigma: T.Scalar,
    epsilon: T.Scalar,
) -> geo.Matrix11:
    """
    Factor representing a Gaussian prior on the inverse range of a landmark

    Args:
        landmark_inverse_range: The current inverse range estimate
        inverse_range_prior: The mean of the inverse range prior
        weight: The weight of the prior
        sigma: The standard deviation of the prior
        epsilon: Small positive value

    Outputs:
        res: 1dof residual of the prior
    """
    prior_diff = landmark_inverse_range - inverse_range_prior
    prior_whitened_diff = weight * prior_diff / (sigma + epsilon)
    return geo.V1(prior_whitened_diff)


def reprojection_delta(
    source_pose: geo.Pose3,
    source_calibration: cam.LinearCameraCal,
    target_pose: geo.Pose3,
    target_calibration: cam.LinearCameraCal,
    source_pixel: geo.Matrix21,
    target_pixel: geo.Matrix21,
    source_inverse_range: T.Scalar,
    epsilon: T.Scalar,
) -> T.Tuple[geo.Matrix21, T.Scalar]:
    """
    Reprojects the landmark into the target camera and returns the delta from the correspondence to
    the reprojection.

    The landmark is specified as a pixel in the source camera and an inverse range; this means the
    landmark is fixed in the source camera and always has residual 0 there (this 0 residual is not
    returned, only the residual in the target camera is returned).
    """
    # Warp source coords into target
    source_cam = cam.PosedCamera(pose=source_pose, calibration=source_calibration)
    target_cam = cam.PosedCamera(pose=target_pose, calibration=target_calibration)

    target_pixel_warped, warp_is_valid = source_cam.warp_pixel(
        pixel=source_pixel,
        inverse_range=source_inverse_range,
        target_cam=target_cam,
        epsilon=epsilon,
    )

    reprojection_error = target_pixel_warped - target_pixel
    return reprojection_error, warp_is_valid


def inverse_range_landmark_reprojection_residual(  # pylint: disable=too-many-arguments
    source_pose: geo.Pose3,
    source_calibration_storage: geo.Matrix41,
    target_pose: geo.Pose3,
    target_calibration_storage: geo.Matrix41,
    source_inverse_range: T.Scalar,
    source_pixel: geo.Matrix21,
    target_pixel: geo.Matrix21,
    weight: T.Scalar,
    gnc_mu: T.Scalar,
    gnc_scale: T.Scalar,
    epsilon: T.Scalar,
) -> geo.Matrix21:
    """
    Return the 2dof residual of reprojecting the landmark into the target camera and comparing
    against the correspondence in the target camera.

    The landmark is specified as a pixel in the source camera and an inverse range; this means the
    landmark is fixed in the source camera and always has residual 0 there (this 0 residual is not
    returned, only the residual in the target camera is returned).

    The norm of the residual is whitened using the Barron noise model.  Whitening each component of
    the reprojection error separately would result in rejecting individual components as outliers.
    Instead, we minimize the whitened norm of the full reprojection error for each point.  See the
    docstring for `NoiseModel.whiten_norm` for more information on this, and the docstring of
    `BarronNoiseModel` for more information on the noise model.

    Args:
        source_pose: The pose of the source camera
        source_calibration_storage: The storage vector of the source (Linear) camera calibration
        target_pose: The pose of the target camera
        target_calibration_storage: The storage vector of the target (Linear) camera calibration
        source_inverse_range: The inverse range of the landmark in the source camera
        source_pixel: The location of the landmark in the source camera
        target_pixel: The location of the correspondence in the target camera
        weight: The weight of the factor
        gnc_mu: The mu convexity parameter for the Barron noise model
        gnc_scale: The scale parameter for the Barron noise model
        epsilon: Small positive value

    Outputs:
        res: 2dof residual of the reprojection
    """
    source_calibration = cam.LinearCameraCal.from_storage(source_calibration_storage.to_flat_list())
    target_calibration = cam.LinearCameraCal.from_storage(target_calibration_storage.to_flat_list())
    reprojection_error, warp_is_valid = reprojection_delta(
        source_pose,
        source_calibration,
        target_pose,
        target_calibration,
        source_pixel,
        target_pixel,
        source_inverse_range,
        epsilon,
    )

    noise_model = BarronNoiseModel(
        alpha=BarronNoiseModel.compute_alpha_from_mu(gnc_mu, epsilon),
        scale=gnc_scale,
        weight=weight * warp_is_valid,
        x_epsilon=epsilon,
        alpha_epsilon=epsilon,
    )

    whitened_residual = noise_model.whiten_norm(reprojection_error)

    return whitened_residual


def generate(output_dir: str) -> None:
    """
    SLAM factors for C++.
    """
    factors_dir = os.path.join(output_dir, "factors")

    codegen.Codegen.function(
        func=inverse_range_landmark_prior_residual, mode=codegen.CodegenMode.CPP
    ).create_with_derivatives(
        which_args=[0],
        name="InverseRangeLandmarkPriorFactor",
        derivative_generation_mode=codegen.DerivativeMode.FULL_LINEARIZATION,
    ).generate_function(
        output_dir=factors_dir, skip_directory_nesting=True
    )

    codegen.Codegen.function(
        func=inverse_range_landmark_reprojection_residual, mode=codegen.CodegenMode.CPP
    ).create_with_derivatives(
        which_args=[0, 2, 4],
        name="InverseRangeLandmarkReprojectionErrorFactor",
        derivative_generation_mode=codegen.DerivativeMode.FULL_LINEARIZATION,
    ).generate_function(
        output_dir=factors_dir, skip_directory_nesting=True
    )
