import os
import tempfile
import textwrap
import collections

from symforce import logger
from symforce import geo
from symforce import cam
from symforce import sympy as sm
from symforce import typing as T
from symforce import path_util
from symforce import python_util
from symforce.codegen import Codegen
from symforce.codegen import CodegenConfig, CppConfig
from symforce.codegen import codegen_util
from symforce.codegen import template_util

from .geo_package_codegen import make_storage_ops_funcs

# Default cam types to generate
DEFAULT_CAM_TYPES = cam.CameraCal.__subclasses__()


def pixel_from_camera_point_with_jacobians(
    self: cam.CameraCal, point: geo.V3, epsilon: T.Scalar
) -> T.Tuple[geo.V2, T.Scalar, geo.M, geo.M]:
    """
    Project a 3D point in the camera frame into 2D pixel coordinates.

    Return:
        pixel: (x, y) coordinate in pixels if valid
        is_valid: 1 if the operation is within bounds else 0
        pixel_D_cal: Derivative of pixel with respect to intrinsic calibration parameters
        pixel_D_point: Derivative of pixel with respect to point

    """
    pixel, is_valid = self.pixel_from_camera_point(point, epsilon)
    pixel_D_cal = pixel.jacobian(self.parameters())
    pixel_D_point = pixel.jacobian(point)
    return pixel, is_valid, pixel_D_cal, pixel_D_point


def camera_ray_from_pixel_with_jacobians(
    self: cam.CameraCal, pixel: geo.V2, epsilon: T.Scalar
) -> T.Tuple[geo.V3, T.Scalar, geo.M, geo.M]:
    """
    Backproject a 2D pixel coordinate into a 3D ray in the camera frame.

    Return:
        camera_ray: The ray in the camera frame (NOT normalized)
        is_valid: 1 if the operation is within bounds else 0
        point_D_cal: Derivative of point with respect to intrinsic calibration parameters
        point_D_pixel: Derivation of point with respect to pixel
    """
    point, is_valid = self.camera_ray_from_pixel(pixel, epsilon)
    point_D_cal = point.jacobian(self.parameters())
    point_D_pixel = point.jacobian(pixel)
    return point, is_valid, point_D_cal, point_D_pixel


def make_camera_funcs(cls: T.Type, config: CodegenConfig) -> T.List[Codegen]:
    """
    Create func spec arguments for common camera operations for the given class.
    """
    camera_ray_from_pixel = None
    try:
        camera_ray_from_pixel = Codegen.function(
            func=cls.camera_ray_from_pixel,
            input_types=[cls, geo.V2, sm.Symbol],
            config=config,
            output_names=["camera_ray", "is_valid"],
            return_key="camera_ray",
            docstring=cam.CameraCal.camera_ray_from_pixel.__doc__,
        )

        camera_ray_from_pixel_with_jacobians_codegen_func = Codegen.function(
            func=camera_ray_from_pixel_with_jacobians,
            input_types=[cls, geo.V2, sm.Symbol],
            config=config,
            output_names=["camera_ray", "is_valid", "point_D_cal", "point_D_pixel"],
            return_key="camera_ray",
            docstring=camera_ray_from_pixel_with_jacobians.__doc__,
        )

    except NotImplementedError:
        # Not all cameras implement backprojection
        pass

    pixel_from_camera_point = Codegen.function(
        func=cls.pixel_from_camera_point,
        config=config,
        input_types=[cls, geo.V3, sm.Symbol],
        output_names=["pixel", "is_valid"],
        return_key="pixel",
        docstring=cam.CameraCal.pixel_from_camera_point.__doc__,
    )

    pixel_from_camera_point_with_jacobians_codegen_func = Codegen.function(
        func=pixel_from_camera_point_with_jacobians,
        config=config,
        input_types=[cls, geo.V3, sm.Symbol],
        output_names=["pixel", "is_valid", "pixel_D_cal", "pixel_D_point"],
        return_key="pixel",
        docstring=pixel_from_camera_point_with_jacobians.__doc__,
    )

    return [
        Codegen.function(
            name="focal_length",
            func=lambda self: self.focal_length,
            input_types=[cls],
            config=config,
            output_names=["focal_length"],
            return_key="focal_length",
            docstring="\nReturn the focal length.",
        ),
        Codegen.function(
            name="principal_point",
            func=lambda self: self.principal_point,
            input_types=[cls],
            config=config,
            output_names=["principal_point"],
            return_key="principal_point",
            docstring="\nReturn the principal point.",
        ),
        pixel_from_camera_point,
        pixel_from_camera_point_with_jacobians_codegen_func,
    ] + (
        [camera_ray_from_pixel, camera_ray_from_pixel_with_jacobians_codegen_func]
        if camera_ray_from_pixel is not None
        else []
    )


def cam_class_data(cls: T.Type, config: CodegenConfig) -> T.Dict[str, T.Any]:
    """
    Data for template generation of this class. Contains all useful info for
    class-specific templates.
    """
    data = Codegen.common_data()
    data["cls"] = cls

    data["specs"] = collections.defaultdict(list)

    for func in make_storage_ops_funcs(cls, config):
        data["specs"]["StorageOps"].append(func)

    for func in make_camera_funcs(cls, config):
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


def generate(config: CodegenConfig, output_dir: str = None) -> str:
    """
    Generate the cam package for the given language.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(
            prefix=f"sf_codegen_{type(config).__name__.lower()}_", dir="/tmp"
        )
        logger.debug(f"Creating temp directory: {output_dir}")

    # Subdirectory for everything we'll generate
    cam_package_dir = os.path.join(output_dir, "sym")
    templates = template_util.TemplateList()

    if isinstance(config, CppConfig):
        logger.info(f'Creating C++ cam package at: "{cam_package_dir}"')
        template_dir = os.path.join(template_util.CPP_TEMPLATE_DIR, "cam_package")

        # First generate the geo package as it's a dependency of the cam package
        from symforce.codegen import geo_package_codegen

        geo_package_codegen.generate(config=config, output_dir=output_dir)

        # Build up templates for each type
        for cls in DEFAULT_CAM_TYPES:
            data = cam_class_data(cls, config=config)

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
        ):

            def supports_camera_ray_from_pixel(cls: T.Type) -> bool:
                try:
                    cls.symbolic("C").camera_ray_from_pixel(geo.V2())
                except NotImplementedError:
                    return False
                else:
                    return True

            templates.add(
                os.path.join(template_dir, "..", "tests", name) + ".jinja",
                os.path.join(output_dir, "tests", name),
                dict(
                    Codegen.common_data(),
                    all_types=DEFAULT_CAM_TYPES,
                    cpp_cam_types=[
                        f"sym::{cls.__name__}<{scalar}>"
                        for cls in DEFAULT_CAM_TYPES
                        for scalar in data["scalar_types"]
                    ],
                    fully_implemented_cpp_cam_types=[
                        f"sym::{cls.__name__}<{scalar}>"
                        for cls in DEFAULT_CAM_TYPES
                        for scalar in data["scalar_types"]
                        if supports_camera_ray_from_pixel(cls)
                    ],
                ),
            )
    else:
        raise NotImplementedError(f'Unknown config type: "{config}"')

    templates.render()

    return output_dir
