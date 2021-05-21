from symforce import codegen
from symforce import logger

from .factor_residuals import (
    relative_pose_prior_residual,
    landmark_prior_residual,
    reprojection_residual,
)


class DynamicBundleAdjustmentProblem:
    """
    The setup is that we have N camera views for which we have poses that we want to refine.
    Camera 0 is taken as the source camera - we don't optimize its pose and treat it as the
    source for all matches. We have feature correspondences from camera 0 into each other camera.
    We put a prior on the relative poses between successive views, and the inverse range of each
    landmark.

    This is called from symforce/test/symforce_bundle_adjustment_example_codegen_test.py to
    actually generate the problem
    """

    def __init__(self) -> None:

        # No setup needed
        pass

    def generate(self, output_dir: str) -> None:
        """
        Generates functions from symbolic expressions
        """

        logger.info("Generating factors for dynamic-size problem")

        namespace = "bundle_adjustment_example"
        codegen.Codegen.function(
            func=landmark_prior_residual, mode=codegen.CodegenMode.CPP
        ).create_with_derivatives(
            which_args=[0],
            name="LandmarkPriorFactor",
            derivative_generation_mode=codegen.DerivativeMode.FULL_LINEARIZATION,
        ).generate_function(
            output_dir=output_dir, namespace=namespace,
        )

        codegen.Codegen.function(
            func=relative_pose_prior_residual, mode=codegen.CodegenMode.CPP
        ).create_with_derivatives(
            which_args=[0, 1],
            name="RelativePosePriorFactor",
            derivative_generation_mode=codegen.DerivativeMode.FULL_LINEARIZATION,
        ).generate_function(
            output_dir=output_dir, namespace=namespace,
        )

        codegen.Codegen.function(
            func=reprojection_residual, mode=codegen.CodegenMode.CPP
        ).create_with_derivatives(
            which_args=[0, 2, 4],
            name="ReprojectionErrorFactor",
            derivative_generation_mode=codegen.DerivativeMode.FULL_LINEARIZATION,
        ).generate_function(
            output_dir=output_dir, namespace=namespace,
        )
