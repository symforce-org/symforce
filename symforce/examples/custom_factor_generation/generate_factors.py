from symforce import codegen
from symforce import logger

from .factor_residuals import custom_between_factor_residual


def generate(output_dir: str) -> None:
    """
    Generate the example custom_between_factor_residual factor

    This is called from symforce/test/symforce_examples_custom_factor_generation_codegen_test.py to
    actually generate the factor
    """
    logger.info("Generating factors")
    namespace = "custom_factor_generation"

    codegen.Codegen.function(
        func=custom_between_factor_residual, config=codegen.CppConfig()
    ).create_with_derivatives(
        which_args=[0, 1],
        name="CustomBetweenFactor",
        derivative_generation_mode=codegen.DerivativeMode.FULL_LINEARIZATION,
    ).generate_function(
        output_dir=output_dir, namespace=namespace
    )
