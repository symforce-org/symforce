# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import codegen
from symforce import logger
from symforce import typing as T

from .factor_residuals import custom_between_factor_residual


def generate(output_dir: T.Openable) -> None:
    """
    Generate the example custom_between_factor_residual factor

    This is called from symforce/test/symforce_examples_custom_factor_generation_codegen_test.py to
    actually generate the factor
    """
    logger.debug("Generating factors")
    namespace = "custom_factor_generation"

    codegen.Codegen.function(
        func=custom_between_factor_residual, config=codegen.CppConfig()
    ).with_linearization(which_args=["nav_T_src", "nav_T_target"]).generate_function(
        output_dir=output_dir, namespace=namespace
    )
