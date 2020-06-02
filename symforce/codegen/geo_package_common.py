import collections
import textwrap

from symforce import geo
from symforce import ops
from symforce import python_util
from symforce import sympy as sm
from symforce import types as T

from .codegen_util import CodegenMode
from .function_codegen import common_data
from .function_codegen import FunctionCodegen

# Default geo types to generate
DEFAULT_TYPES = (geo.Rot2, geo.Pose2, geo.Rot3, geo.Pose3)


def make_storage_ops_funcs(cls):
    # type: (T.Type) -> T.List[FunctionCodegen]
    """
    Create func spec arguments for storage_ops on the given class.
    """
    storage_vec = geo.M(range(ops.StorageOps.storage_dim(cls)))
    return [
        FunctionCodegen("ToStorage", ops.StorageOps.to_storage, [cls], storage_vec),
        FunctionCodegen(
            "FromStorage", lambda vec: ops.StorageOps.from_storage(cls, vec), [storage_vec], cls
        ),
    ]


def make_group_ops_funcs(cls):
    # type: (T.Type) -> T.List[FunctionCodegen]
    """
    Create func spec arguments for group ops on the given class.
    """
    return [
        FunctionCodegen("Identity", lambda: ops.GroupOps.identity(cls), [], cls,),
        FunctionCodegen("Inverse", ops.GroupOps.inverse, [cls], cls,),
        FunctionCodegen("Compose", ops.GroupOps.compose, [cls, cls], cls),
        FunctionCodegen("Between", ops.GroupOps.between, [cls, cls], cls,),
    ]


def make_lie_group_ops_funcs(cls):
    # type: (T.Type) -> T.List[FunctionCodegen]
    """
    Create func spec arguments for lie group ops on the given class.
    """
    tangent_vec = geo.M(range(ops.LieGroupOps.tangent_dim(cls)))
    return [
        FunctionCodegen(
            "FromTangent",
            lambda vec, epsilon: ops.LieGroupOps.from_tangent(cls, vec, epsilon),
            [tangent_vec, sm.Symbol],
            cls,
        ),
        FunctionCodegen("ToTangent", ops.LieGroupOps.to_tangent, [cls, sm.Symbol], tangent_vec,),
        FunctionCodegen("Retract", ops.LieGroupOps.retract, [cls, tangent_vec, sm.Symbol], cls),
        FunctionCodegen(
            "LocalCoordinates",
            ops.LieGroupOps.local_coordinates,
            [cls, cls, sm.Symbol],
            tangent_vec,
        ),
    ]


def class_data(cls, mode):
    # type: (T.Type, CodegenMode) -> T.Dict[str, T.Any]
    """
    Data for template generation of this class. Contains all useful info for
    class-specific templates.
    """
    data = common_data()
    data["cls"] = cls

    data["specs"] = collections.defaultdict(list)

    for func in make_storage_ops_funcs(cls):
        func.gen_code(mode=mode)
        data["specs"]["StorageOps"].append(func)

    for func in make_group_ops_funcs(cls):
        func.gen_code(mode=mode)
        data["specs"]["GroupOps"].append(func)

    for func in make_lie_group_ops_funcs(cls):
        func.gen_code(mode=mode)
        data["specs"]["LieGroupOps"].append(func)

    data["doc"] = textwrap.dedent(cls.__doc__).strip() if cls.__doc__ else ""
    data["is_lie_group"] = hasattr(cls, "from_tangent")

    return data
