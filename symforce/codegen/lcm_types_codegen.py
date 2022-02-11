from symforce import python_util
from symforce import typing as T
from symforce.codegen import cam_package_codegen


def lcm_symforce_types_data() -> T.Dict[str, T.Any]:
    """
    Returns data for template generation with lcm_templates/symforce_types.lcm.jinja.
    """
    return dict(
        python_util=python_util, camera_cal_class_names=cam_package_codegen.camera_cal_class_names()
    )
