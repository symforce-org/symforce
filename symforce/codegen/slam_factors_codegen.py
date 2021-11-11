import os

from symforce import cam
from symforce import codegen
from symforce import geo
from symforce import sympy as sm
from symforce import typing as T
from symforce.opt.noise_models import BarronNoiseModel


def inverse_range_landmark_prior_residual(
    landmark_inverse_range: T.Scalar,
    inverse_range_prior: T.Scalar,
    weight: T.Scalar,
    sigma: T.Scalar,
    epsilon: T.Scalar,
) -> geo.Vector1:
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
    source_pixel: geo.Vector2,
    target_pixel: geo.Vector2,
    source_inverse_range: T.Scalar,
    epsilon: T.Scalar,
) -> T.Tuple[geo.Vector2, T.Scalar]:
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


def inverse_range_landmark_reprojection_error_residual(
    source_pose: geo.Pose3,
    source_calibration_storage: geo.Vector4,
    target_pose: geo.Pose3,
    target_calibration_storage: geo.Vector4,
    source_inverse_range: T.Scalar,
    source_pixel: geo.Vector2,
    target_pixel: geo.Vector2,
    weight: T.Scalar,
    gnc_mu: T.Scalar,
    gnc_scale: T.Scalar,
    epsilon: T.Scalar,
) -> geo.Vector2:
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


def spherical_reprojection_delta(
    source_pose: geo.Pose3,
    target_pose: geo.Pose3,
    target_calibration_storage: geo.Vector9,
    source_inverse_range: T.Scalar,
    p_camera_source: geo.Vector3,
    target_pixel: geo.Vector2,
    epsilon: T.Scalar,
) -> T.Tuple[geo.Vector2, T.Scalar]:
    """
    Reprojects the landmark ray into the target spherical camera and returns the delta between the
    correspondence and the reprojection.

    The landmark is specified as a 3D point or ray (will be normalized) in the source spherical
    camera; this means the landmark is fixed in the source camera and always has residual 0 there
    (this 0 residual is not returned, only the residual in the target camera is returned).

    Args:
        source_pose: The pose of the source camera
        target_pose: The pose of the target camera
        target_calibration_storage: The storage vector of the target spherical camera calibration
        source_inverse_range: The inverse range of the landmark in the source camera
        p_camera_source: The location of the landmark in the source camera coordinate, will be normalized
        target_pixel: The location of the correspondence in the target camera
        epsilon: Small positive value

    Outputs:
        res: 2dof reprojection delta
        valid: is valid projection or not
    """
    nav_T_target_cam = target_pose
    nav_T_source_cam = source_pose
    p_camera_source_unit_ray = p_camera_source / p_camera_source.norm(epsilon)

    p_cam_target = nav_T_target_cam.R.inverse() * (
        nav_T_source_cam.R * p_camera_source_unit_ray
        + (nav_T_source_cam.t - nav_T_target_cam.t) * source_inverse_range
    )

    target_cam = cam.SphericalCameraCal.from_storage(target_calibration_storage.to_flat_list())
    target_pixel_reprojection, is_valid = target_cam.pixel_from_camera_point(
        p_cam_target, epsilon=epsilon
    )
    reprojection_error = target_pixel_reprojection - target_pixel

    return reprojection_error, is_valid


def inverse_range_landmark_spherical_camera_reprojection_error_residual(
    source_pose: geo.Pose3,
    target_pose: geo.Pose3,
    target_calibration_storage: geo.Vector9,
    source_inverse_range: T.Scalar,
    p_camera_source: geo.Vector3,
    target_pixel: geo.Vector2,
    weight: T.Scalar,
    gnc_mu: T.Scalar,
    gnc_scale: T.Scalar,
    epsilon: T.Scalar,
) -> geo.Vector2:
    """
    Return the 2dof residual of reprojecting the landmark ray into the target spherical camera and comparing
    it against the correspondence.

    The landmark is specified as a camera point in the source camera with an inverse range; this means the
    landmark is fixed in the source camera and always has residual 0 there (this 0 residual is not
    returned, only the residual in the target camera is returned).

    The norm of the residual is whitened using the Barron noise model.  Whitening each component of
    the reprojection error separately would result in rejecting individual components as outliers.
    Instead, we minimize the whitened norm of the full reprojection error for each point.  See the
    docstring for `NoiseModel.whiten_norm` for more information on this, and the docstring of
    `BarronNoiseModel` for more information on the noise model.

    Args:
        source_pose: The pose of the source camera
        target_pose: The pose of the target camera
        target_calibration_storage: The storage vector of the target spherical camera calibration
        source_inverse_range: The inverse range of the landmark in the source camera
        p_camera_source: The location of the landmark in the source camera coordinate, will be normalized
        target_pixel: The location of the correspondence in the target camera
        weight: The weight of the factor
        gnc_mu: The mu convexity parameter for the Barron noise model
        gnc_scale: The scale parameter for the Barron noise model
        epsilon: Small positive value

    Outputs:
        res: 2dof whiten residual of the reprojection
    """
    reprojection_error, is_valid = spherical_reprojection_delta(
        source_pose,
        target_pose,
        target_calibration_storage,
        source_inverse_range,
        p_camera_source,
        target_pixel,
        epsilon,
    )

    # Compute whitened residual with a noise model.
    alpha = BarronNoiseModel.compute_alpha_from_mu(mu=gnc_mu, epsilon=epsilon)
    noise_model = BarronNoiseModel(
        alpha=alpha,
        scale=gnc_scale,
        weight=weight * is_valid,
        x_epsilon=epsilon,
        alpha_epsilon=epsilon,
    )

    whitened_residual = noise_model.whiten_norm(reprojection_error)

    return whitened_residual


def generate(output_dir: str, config: codegen.CodegenConfig = None) -> None:
    """
    Generate the SLAM package for the given language.

    Args:
        output_dir: Directory to generate outputs into
        config: CodegenConfig, defaults to the default C++ config
    """
    # Subdirectory for everything we'll generate
    factors_dir = os.path.join(output_dir, "factors")

    # default config to CppConfig
    if config is None:
        config = codegen.CppConfig()

    codegen.Codegen.function(
        func=inverse_range_landmark_prior_residual, config=config
    ).with_linearization(which_args=[0]).generate_function(
        output_dir=factors_dir, skip_directory_nesting=True
    )

    codegen.Codegen.function(
        func=inverse_range_landmark_reprojection_error_residual, config=config
    ).with_linearization(which_args=[0, 2, 4]).generate_function(
        output_dir=factors_dir, skip_directory_nesting=True
    )

    codegen.Codegen.function(
        func=spherical_reprojection_delta,
        config=config,
        output_names=["reprojection_delta", "is_valid"],
    ).generate_function(output_dir=factors_dir, skip_directory_nesting=True)

    codegen.Codegen.function(
        func=inverse_range_landmark_spherical_camera_reprojection_error_residual, config=config
    ).with_linearization(which_args=[0, 1, 3]).generate_function(
        output_dir=factors_dir, skip_directory_nesting=True
    )
