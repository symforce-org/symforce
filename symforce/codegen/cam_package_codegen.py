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
DEFAULT_CAM_TYPES = (
    cam.LinearCameraCal,
    cam.EquidistantEpipolarCameraCal,
    cam.ATANCameraCal,
)


def make_camera_funcs(cls: T.Type, mode: codegen_util.CodegenMode) -> T.List[Codegen]:
    """
    Create func spec arguments for common camera operations for the given class.
    """
    return [
        Codegen.function(
            name="FocalLength",
            func=lambda self: self.focal_length,
            input_types=[cls],
            mode=mode,
            output_names=["focal_length"],
            return_key="focal_length",
            docstring="\nReturn the focal length.",
        ),
        Codegen.function(
            name="PrincipalPoint",
            func=lambda self: self.principal_point,
            input_types=[cls],
            mode=mode,
            output_names=["principal_point"],
            return_key="principal_point",
            docstring="\nReturn the principal point.",
        ),
        Codegen.function(
            name="PixelFromCameraPoint",
            func=cls.pixel_from_camera_point,
            mode=mode,
            input_types=[cls, geo.V3, sm.Symbol],
            output_names=["pixel", "is_valid"],
            return_key="pixel",
            docstring=cam.CameraCal.pixel_from_camera_point.__doc__,
        ),
        Codegen.function(
            name="CameraRayFromPixel",
            func=cls.camera_ray_from_pixel,
            input_types=[cls, geo.V2, sm.Symbol],
            mode=mode,
            output_names=["camera_ray", "is_valid"],
            return_key="camera_ray",
            docstring=cam.CameraCal.camera_ray_from_pixel.__doc__,
        ),
    ]


def cam_class_data(cls: T.Type, mode: CodegenMode) -> T.Dict[str, T.Any]:
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


def class_template_data(
    cls: T.Type, functions_to_doc: T.Sequence["function"]
) -> T.Dict[str, T.Any]:
    data = Codegen.common_data()
    data["doc"] = dict()
    data["doc"]["cls"] = textwrap.dedent(cls.__doc__).strip()  # type: ignore
    for func in functions_to_doc:
        if func.__doc__ is not None:
            data["doc"][func.__name__] = textwrap.dedent(func.__doc__)
        else:
            data["doc"][func.__name__] = None
    return data


def camera_data() -> T.Dict[str, T.Any]:
    functions_to_doc = [
        cam.Camera.pixel_from_camera_point,
        cam.Camera.camera_ray_from_pixel,
        cam.Camera.maybe_check_in_view,
        cam.Camera.in_view,
    ]
    return class_template_data(cam.Camera, functions_to_doc)


def posed_camera_data() -> T.Dict[str, T.Any]:
    functions_to_doc = [
        cam.PosedCamera.pixel_from_global_point,
        cam.PosedCamera.global_point_from_pixel,
        cam.PosedCamera.warp_pixel,
    ]
    return class_template_data(cam.PosedCamera, functions_to_doc)


def generate(mode: CodegenMode, output_dir: str = None) -> str:
    """
    Generate the cam package for the given language.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix=f"sf_codegen_{mode.name}_", dir="/tmp")
        logger.debug(f"Creating temp directory: {output_dir}")

    # Subdirectory for everything we'll generate
    cam_package_dir = os.path.join(output_dir, "sym")
    templates = template_util.TemplateList()

    if mode == CodegenMode.CPP:
        logger.info(f'Creating C++ cam package at: "{cam_package_dir}"')
        template_dir = os.path.join(template_util.CPP_TEMPLATE_DIR, "cam_package")

        # First generate the geo package as it's a dependency of the cam package
        from symforce.codegen import geo_package_codegen

        geo_package_codegen.generate(mode=CodegenMode.CPP, output_dir=output_dir)

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

        # Add Camera and PosedCamera
        templates.add(
            os.path.join(template_dir, "camera.h.jinja"),
            os.path.join(cam_package_dir, "camera.h"),
            camera_data(),
        )
        templates.add(
            os.path.join(template_dir, "posed_camera.h.jinja"),
            os.path.join(cam_package_dir, "posed_camera.h"),
            posed_camera_data(),
        )

        # Test example
        for name in (
            "cam_package_cpp_test.cc",
            "cam_function_codegen_cpp_test.cc",
            "Makefile",
        ):
            templates.add(
                os.path.join(template_dir, "..", "example", name) + ".jinja",
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
                    symforce_include_dir=os.path.join(CURRENT_DIR, "../../"),
                    lib_dir=os.path.join(output_dir, "example"),
                ),
            )
    else:
        raise NotImplementedError(f'Unknown mode: "{mode}"')

    templates.render()

    return output_dir
