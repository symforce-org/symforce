from symforce import cam
from symforce import python_util
from symforce import typing as T


def lcm_symforce_types_data() -> T.Dict[str, T.Any]:
    """
    Returns data for template generation with lcm_templates/symforce_types.lcm.jinja.
    """
    enums = [
        python_util.camelcase_to_screaming_snakecase(cal_cls.__name__)
        for cal_cls in cam.CameraCal.__subclasses__()
    ]
    enums.sort()
    return dict(camera_cal_enum_names=enums)
