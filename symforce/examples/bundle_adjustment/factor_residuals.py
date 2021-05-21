from symforce import cam
from symforce import geo
from symforce import sympy as sm
from symforce import types as T
from symforce.opt.noise_models import BarronNoiseModel


def relative_pose_prior_residual(
    nav_T_src: geo.Pose3,
    nav_T_target: geo.Pose3,
    prior_rotation: geo.Rot3,
    prior_translation: geo.Matrix31,
    prior_weight: T.Scalar,
    prior_sigmas: geo.Matrix61,
    epsilon: T.Scalar,
) -> geo.Matrix61:
    """
    Return the 6dof residual on the relative pose between the given two views. Compares
    the relative pose between the optimized poses to the relative pose between the priors.

    Args:
        nav_T_src: Pose of source camera
        nav_T_target: Pose of target camera
        prior_rotation: The rotation component of the prior
        prior_translation: The translation component of the prior
        prior_sigmas: Diagonal standard deviations of the prior (i.e. sqrt(diag(cov)))
    """
    # Take local coordinates individually for translation and rotation.
    # Doing it on the Se3 object properly incurs unnecessary complexity.
    target_t_src_actual = nav_T_target.t.between(nav_T_src.t)
    target_t_src_desired = prior_translation
    position_error = target_t_src_actual.local_coordinates(target_t_src_desired, epsilon=epsilon)

    target_R_src_actual = nav_T_target.R.between(nav_T_src.R)
    target_R_src_desired = prior_rotation
    rotation_error = target_R_src_actual.local_coordinates(target_R_src_desired, epsilon=epsilon)

    # Whiten the error by the sigmas of the target pose prior, plus a unitless weight
    weight = prior_weight
    target_pose_sigmas = prior_sigmas
    residual = geo.M(list(rotation_error) + list(position_error))

    # NOTE(aaron): sqrt(weight) is safe because weight is a hyperparameter, adding epsilon
    # breaks the weight==0 case
    whitened_residual = geo.Vector6(
        [
            sm.sqrt(weight) * err / (sigma + epsilon)
            for err, sigma in zip(residual.to_flat_list(), target_pose_sigmas.to_flat_list())
        ]
    )

    return whitened_residual


def landmark_prior_residual(
    landmark: T.Scalar,
    inverse_range_prior: T.Scalar,
    weight: T.Scalar,
    sigma: T.Scalar,
    epsilon: T.Scalar,
) -> geo.Matrix11:
    prior_diff = landmark - inverse_range_prior
    prior_whitened_diff = weight * prior_diff / (sigma + epsilon)
    return geo.V1(prior_whitened_diff)


def reprojection_delta(
    source_pose: geo.Pose3,
    source_calibration: cam.LinearCameraCal,
    target_pose: geo.Pose3,
    target_calibration: cam.LinearCameraCal,
    source_coords: geo.Matrix21,
    target_coords: geo.Matrix21,
    source_inv_range: T.Scalar,
    epsilon: T.Scalar,
) -> T.Tuple[geo.Matrix21, T.Scalar]:
    """
    Return the delta from reprojecting the landmark into the target camera
    from its correspondence.
    """
    # Warp source coords into target
    source_cam = cam.PosedCamera(pose=source_pose, calibration=source_calibration)
    target_cam = cam.PosedCamera(pose=target_pose, calibration=target_calibration)

    target_coords_warped, warp_is_valid = source_cam.warp_pixel(
        pixel=source_coords, inverse_range=source_inv_range, target_cam=target_cam, epsilon=epsilon,
    )

    reprojection_error = target_coords_warped - target_coords
    return reprojection_error, warp_is_valid


def reprojection_residual(  # pylint: disable=too-many-arguments
    source_pose: geo.Pose3,
    source_calibration_storage: geo.Matrix41,
    target_pose: geo.Pose3,
    target_calibration_storage: geo.Matrix41,
    source_inv_range: T.Scalar,
    source_coords: geo.Matrix21,
    target_coords: geo.Matrix21,
    weight: T.Scalar,
    epsilon: T.Scalar,
    gnc_mu: T.Scalar,
    gnc_scale: T.Scalar,
) -> geo.Matrix21:
    """
    Return the 2dof residual of reprojecting the landmark into the target
    camera.
    """
    source_calibration = cam.LinearCameraCal.from_storage(source_calibration_storage.to_flat_list())
    target_calibration = cam.LinearCameraCal.from_storage(target_calibration_storage.to_flat_list())
    reprojection_error, warp_is_valid = reprojection_delta(
        source_pose,
        source_calibration,
        target_pose,
        target_calibration,
        source_coords,
        target_coords,
        source_inv_range,
        epsilon,
    )

    noise_model = BarronNoiseModel(
        alpha=BarronNoiseModel.compute_alpha_from_mu(gnc_mu, epsilon),
        scale=gnc_scale,
        weight=weight * warp_is_valid,
        x_epsilon=epsilon,
        alpha_epsilon=epsilon,
    )

    # Whitening each component of the reprojection error separately results in rejecting
    # individual components as outliers. Instead, we minimize the whitened norm of the full
    # reprojection error for each point.  See the docstring for `NoiseModel.whiten_norm` for
    # more information.
    whitened_residual = noise_model.whiten_norm(reprojection_error)

    return whitened_residual
