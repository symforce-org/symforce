from __future__ import absolute_import

import tempfile

from symforce import logger

from .codegen_util import CodegenMode


def generate(output_dir, mode):
    """
    Generate the geo package for the given language.

    TODO(hayk): Take scalar_type list here.

    Args:
        output_dir (str or None):
        mode (CodegenMode):
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="sf_codegen_{}_".format(self.name), dir="/tmp")
        logger.debug("Creating temp directory: {}".format(output_dir))

    if mode == CodegenMode.PYTHON2:
        from symforce.codegen.python import python_geo_package

        codegen_data = python_geo_package.generate(output_dir)

    elif mode == CodegenMode.CPP:
        from symforce.codegen.cpp import cpp_geo_package

        codegen_data = cpp_geo_package.generate(output_dir)
    else:
        raise NotImplementedError('Unknown mode: "{}"'.format(mode))

    return output_dir
