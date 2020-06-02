from __future__ import absolute_import

import tempfile

from symforce import logger
from symforce import types as T

from .codegen_util import CodegenMode


def generate(mode, output_dir=None):
    # type: (CodegenMode, str) -> str
    """
    Generate the geo package for the given language.

    TODO(hayk): Take scalar_type list here.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="sf_codegen_{}_".format(mode.name), dir="/tmp")
        logger.debug("Creating temp directory: {}".format(output_dir))

    if mode == CodegenMode.PYTHON2:
        from symforce.codegen.python import python_geo_package

        python_geo_package.generate(output_dir)

    elif mode == CodegenMode.CPP:
        from symforce.codegen.cpp import cpp_geo_package

        cpp_geo_package.generate(output_dir)
    else:
        raise NotImplementedError('Unknown mode: "{}"'.format(mode))

    return output_dir
