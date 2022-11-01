# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import collections
import pathlib
import tempfile
import textwrap

import numpy as np

import symforce.symbolic as sf
from symforce import logger
from symforce import python_util
from symforce import typing as T
from symforce.codegen import Codegen
from symforce.codegen import CodegenConfig
from symforce.codegen import CppConfig
from symforce.codegen import PythonConfig
from symforce.codegen import template_util
from symforce.codegen.ops_codegen_util import make_group_ops_funcs
from symforce.codegen.ops_codegen_util import make_lie_group_ops_funcs

# Default cam types to generate
DEFAULT_CAM_TYPES = sf.CameraCal.__subclasses__()


def camera_cal_class_names() -> T.List[str]:
    """
    Returns a sorted list of the CameraCal subclass names.
    """
    class_names = [cam_cal_cls.__name__ for cam_cal_cls in sf.CameraCal.__subclasses__()]
    class_names.sort()
    return class_names


def pixel_from_camera_point_with_jacobians(
    self: sf.CameraCal, point: sf.V3, epsilon: sf.Scalar
) -> T.Tuple[sf.V2, sf.Scalar, sf.M, sf.M]:
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
    self: sf.CameraCal, pixel: sf.V2, epsilon: sf.Scalar
) -> T.Tuple[sf.V3, sf.Scalar, sf.M, sf.M]:
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
            input_types=[cls, sf.V2, sf.Symbol],
            config=config,
            output_names=["camera_ray", "is_valid"],
            return_key="camera_ray",
            docstring=sf.CameraCal.camera_ray_from_pixel.__doc__,
        )

        camera_ray_from_pixel_with_jacobians_codegen_func = Codegen.function(
            func=camera_ray_from_pixel_with_jacobians,
            input_types=[cls, sf.V2, sf.Symbol],
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
        input_types=[cls, sf.V3, sf.Symbol],
        output_names=["pixel", "is_valid"],
        return_key="pixel",
        docstring=sf.CameraCal.pixel_from_camera_point.__doc__,
    )

    pixel_from_camera_point_with_jacobians_codegen_func = Codegen.function(
        func=pixel_from_camera_point_with_jacobians,
        config=config,
        input_types=[cls, sf.V3, sf.Symbol],
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

    for func in make_group_ops_funcs(cls, config):
        data["specs"]["GroupOps"].append(func)

    for func in make_lie_group_ops_funcs(cls, config):
        data["specs"]["LieGroupOps"].append(func)

    for func in make_camera_funcs(cls, config):
        data["specs"]["CameraOps"].append(func)

    data["storage_order"] = cls.storage_order()

    data["doc"] = textwrap.dedent(cls.__doc__).strip() if cls.__doc__ else ""

    return data


def class_template_data(
    cls: T.Type, functions_to_doc: T.Sequence["function"]
) -> T.Dict[str, T.Any]:
    data = Codegen.common_data()
    data["doc"] = {}
    data["doc"]["cls"] = textwrap.dedent(cls.__doc__).strip()  # type: ignore
    for func in functions_to_doc:
        if func.__doc__ is not None:
            data["doc"][func.__name__] = textwrap.dedent(func.__doc__)
        else:
            data["doc"][func.__name__] = None
    return data


def camera_data() -> T.Dict[str, T.Any]:
    functions_to_doc = [
        sf.Camera.pixel_from_camera_point,
        sf.Camera.camera_ray_from_pixel,
        sf.Camera.maybe_check_in_view,
        sf.Camera.in_view,
    ]
    return class_template_data(sf.Camera, functions_to_doc)


def posed_camera_data() -> T.Dict[str, T.Any]:
    functions_to_doc = [
        sf.PosedCamera.pixel_from_global_point,
        sf.PosedCamera.global_point_from_pixel,
        sf.PosedCamera.warp_pixel,
    ]
    return class_template_data(sf.PosedCamera, functions_to_doc)


_DISTORTION_COEFF_VALS: T.Dict[str, T.Dict[str, T.Any]] = {
    sf.ATANCameraCal.__name__: {"omega": 0.5},
    sf.DoubleSphereCameraCal.__name__: {"xi": 5.1, "alpha": -6.2},
    sf.PolynomialCameraCal.__name__: {
        "critical_undistorted_radius": np.pi / 3,
        "distortion_coeffs": [0.035, -0.025, 0.0070],
    },
    sf.SphericalCameraCal.__name__: {
        "critical_theta": np.pi,
        "distortion_coeffs": [0.035, -0.025, 0.0070, -0.0015],
    },
}

CamCls = T.TypeVar("CamCls", bound=sf.CameraCal)


def cam_cal_from_points(
    cam_cls: T.Type[CamCls],
    focal_length: T.Sequence[sf.Scalar],
    principal_point: T.Sequence[sf.Scalar],
) -> CamCls:
    """
    Returns an instance of cam_cls with given focal_length and prinicpal_point.
    The purpose of this function is to make it easy to construct camera cals of various
    types without worrying what the extra arguments need to be.
    """
    return cam_cls(
        focal_length=focal_length,
        principal_point=principal_point,
        **_DISTORTION_COEFF_VALS.get(cam_cls.__name__, {}),
    )


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
    cam_package_dir = pathlib.Path(output_dir, "sym")
    template_dir = config.template_dir()
    templates = template_util.TemplateList(template_dir)

    if isinstance(config, PythonConfig):
        logger.debug(f'Creating Python package at: "{cam_package_dir}"')

        # First generate the geo package as it's a dependency of the cam package
        from symforce.codegen import geo_package_codegen

        geo_package_codegen.generate(config=config, output_dir=output_dir)

        # Build up templates for each type

        for cls in DEFAULT_CAM_TYPES:
            data = cam_class_data(cls, config=config)

            for base_dir, relative_path in (
                ("cam_package", "CLASS.py"),
                ("cam_package", "ops/CLASS/camera_ops.py"),
                ("cam_package", "ops/CLASS/__init__.py"),
                (".", "ops/CLASS/group_ops.py"),
                (".", "ops/CLASS/lie_group_ops.py"),
            ):
                template_path = pathlib.Path(base_dir, relative_path + ".jinja")
                output_path = cam_package_dir / relative_path.replace(
                    "CLASS", python_util.camelcase_to_snakecase(cls.__name__)
                )
                templates.add(template_path, data, output_path=output_path)

        # Package init
        # NOTE(brad): We already do this in geo_package_codegen.py. We need it there in case we
        # are generating the geo package but not the cam package. But if we are generating the
        # cam package, we need to make sure it also includes the cam types. So, we overwrite the
        # one generated by the geo package to include the came types.
        templates.add(
            template_path=pathlib.Path("geo_package", "__init__.py.jinja"),
            data=dict(
                Codegen.common_data(),
                all_types=list(geo_package_codegen.DEFAULT_GEO_TYPES) + list(DEFAULT_CAM_TYPES),
                numeric_epsilon=sf.numeric_epsilon,
            ),
            output_path=cam_package_dir
            / ("_init.py" if config.namespace_package else "__init__.py"),
        )

        for name in ("cam_package_python_test.py",):
            templates.add(
                template_path=pathlib.Path("tests", name + ".jinja"),
                output_path=pathlib.Path(output_dir, "tests", name),
                data=dict(
                    Codegen.common_data(),
                    all_types=DEFAULT_CAM_TYPES,
                    cam_cal_from_points=cam_cal_from_points,
                    _DISTORTION_COEFF_VALS=_DISTORTION_COEFF_VALS,
                ),
            )

    elif isinstance(config, CppConfig):
        logger.debug(f'Creating C++ cam package at: "{cam_package_dir}"')
        template_dir = config.template_dir()

        # First generate the geo package as it's a dependency of the cam package
        from symforce.codegen import geo_package_codegen

        geo_package_codegen.generate(config=config, output_dir=output_dir)

        # Build up templates for each type
        for cls in DEFAULT_CAM_TYPES:
            data = cam_class_data(cls, config=config)

            for base_dir, relative_path in (
                ("cam_package", "CLASS.h"),
                ("cam_package", "CLASS.cc"),
                (".", "ops/CLASS/storage_ops.h"),
                (".", "ops/CLASS/storage_ops.cc"),
                (".", "ops/CLASS/group_ops.h"),
                (".", "ops/CLASS/group_ops.cc"),
                (".", "ops/CLASS/lie_group_ops.h"),
                (".", "ops/CLASS/lie_group_ops.cc"),
            ):
                template_path = pathlib.Path(base_dir, relative_path + ".jinja")
                output_path = cam_package_dir / relative_path.replace(
                    "CLASS", python_util.camelcase_to_snakecase(cls.__name__)
                )
                templates.add(template_path, data, output_path=output_path)

        # Add Camera and PosedCamera
        templates.add(
            template_path=pathlib.Path("cam_package", "camera.h.jinja"),
            output_path=cam_package_dir / "camera.h",
            data=camera_data(),
        )
        templates.add(
            template_path=pathlib.Path("cam_package") / "posed_camera.h.jinja",
            output_path=cam_package_dir / "posed_camera.h",
            data=posed_camera_data(),
        )

        # Test example
        for name in (
            "cam_package_cpp_test.cc",
            "cam_function_codegen_cpp_test.cc",
        ):

            def supports_camera_ray_from_pixel(cls: T.Type) -> bool:
                try:
                    cls.symbolic("C").camera_ray_from_pixel(sf.V2())
                except NotImplementedError:
                    return False
                else:
                    return True

            templates.add(
                template_path=pathlib.Path("tests", name + ".jinja"),
                output_path=pathlib.Path(output_dir, "tests", name),
                data=dict(
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
