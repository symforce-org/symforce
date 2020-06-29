from __future__ import absolute_import

import os
import tempfile
import textwrap
import collections

from symforce import logger
from symforce import geo
from symforce import cam
from symforce import sympy as sm
from symforce import types as T
from symforce import python_util
from symforce.codegen import Codegen
from symforce.codegen import CodegenMode
from symforce.codegen import codegen_util
from symforce.codegen import template_util

from .geo_package_codegen import make_storage_ops_funcs

CURRENT_DIR = os.path.dirname(__file__)
# Default cam types to generate
DEFAULT_CAM_TYPES = (cam.LinearCameraCal,)


def make_camera_funcs(cls, mode):
    # type: (T.Type, codegen_util.CodegenMode) -> T.List[Codegen]
    """
    Create func spec arguments for common camera operations for the given class.
    """
    return [
        Codegen.function(
            name="PixelCoordsFromCameraPoint",
            func=cls.pixel_coords_from_camera_point,
            mode=mode,
            input_types=[cls, geo.V3, sm.Symbol],
            output_names=["pixel_coords", "is_valid"],
            return_key="pixel_coords",
            docstring=cam.CameraCal.pixel_coords_from_camera_point.__doc__,
        ),
        Codegen.function(
            name="CameraRayFromPixelCoords",
            func=cls.camera_ray_from_pixel_coords,
            input_types=[cls, geo.V2, sm.Symbol],
            mode=mode,
            output_names=["camera_ray", "is_valid"],
            return_key="camera_ray",
            docstring=cam.CameraCal.camera_ray_from_pixel_coords.__doc__,
        ),
    ]


def cam_class_data(cls, mode):
    # type: (T.Type, CodegenMode) -> T.Dict[str, T.Any]
    """
    Data for template generation of this class. Contains all useful info for
    class-specific templates.
    """
    data = Codegen.common_data()
    data["cls"] = cls

    data["specs"] = collections.defaultdict(list)

    for func in make_storage_ops_funcs(cls, mode):
        data["specs"]["StorageOps"].append(func)

    for func in make_camera_funcs(cls, mode):
        data["specs"]["CameraOps"].append(func)

    data["doc"] = textwrap.dedent(cls.__doc__).strip() if cls.__doc__ else ""

    return data


def generate(mode, output_dir=None):
    # type: (CodegenMode, str) -> str
    """
    Generate the cam package for the given language.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="sf_codegen_{}_".format(mode.name), dir="/tmp")
        logger.debug("Creating temp directory: {}".format(output_dir))

    # Subdirectory for everything we'll generate
    cam_package_dir = os.path.join(output_dir, "cam")
    templates = template_util.TemplateList()

    if mode == CodegenMode.CPP:
        logger.info('Creating C++ cam package at: "{}"'.format(cam_package_dir))
        template_dir = os.path.join(template_util.CPP_TEMPLATE_DIR, "cam_package")

        # First generate the geo package as it's a dependency of the cam package
        from symforce.codegen import geo_package_codegen

        geo_package_codegen.generate(CodegenMode.CPP, output_dir)

        # Build up templates for each type
        for cls in DEFAULT_CAM_TYPES:
            data = cam_class_data(cls, mode=CodegenMode.CPP)

            for path in (
                "CLASS.h",
                "CLASS.cc",
                "ops/CLASS/storage_ops.h",
                "ops/CLASS/storage_ops.cc",
            ):
                template_path = os.path.join(template_dir, path) + ".jinja"
                output_path = os.path.join(cam_package_dir, path).replace(
                    "CLASS", python_util.camelcase_to_snakecase(cls.__name__)
                )
                templates.add(template_path, output_path, data)

        # Test example
        for name in (
            "cam_package_cpp_test.cc",
            "cam_function_codegen_cpp_test.cc",
            "Makefile",
        ):
            templates.add(
                os.path.join(template_dir, "example", name) + ".jinja",
                os.path.join(output_dir, "example", name),
                dict(
                    Codegen.common_data(),
                    all_types=DEFAULT_CAM_TYPES,
                    include_dir=output_dir,
                    eigen_include_dir=os.path.realpath(
                        os.path.join(
                            CURRENT_DIR, "***REMOVED***/include/eigen3/"
                        )
                    ),
                    lib_dir=os.path.join(output_dir, "example"),
                ),
            )
    else:
        raise NotImplementedError('Unknown mode: "{}"'.format(mode))

    templates.render()

    return output_dir
