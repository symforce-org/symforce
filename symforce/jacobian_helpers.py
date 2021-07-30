from symforce import geo
from symforce import ops
from symforce import types as T


def tangent_jacobians(expr: T.Element, args: T.Sequence[T.Element]) -> T.List[geo.Matrix]:
    """
    Compute jacobians of expr, a Lie Group element which is a function of the Lie Group elements in
    args.  Jacobians are derivatives in the tangent space of expr with respect to changes in the
    tangent space of the arg, as opposed to jacobians of the storage of either which could be
    trivially computed with geo.Matrix.jacobian or sm.Expr.diff

    Args:
        expr: The final expression that should be differentiated
        args: Sequence of variables (can be Lie Group elements) to differentiate with respect to

    Returns:
        The jacobian expr_D_arg for each arg in args, where each expr_D_arg is of shape MxN, with M
        the tangent space dimension of expr and N the tangent space dimension of arg
    """
    jacobians = []

    # Compute jacobians in the space of the storage, then chain rule on the left and right sides
    # to get jacobian wrt the tangent space of both the arg and the result
    expr_storage = geo.M(ops.StorageOps.to_storage(expr))
    expr_tangent_D_storage = ops.LieGroupOps.tangent_D_storage(expr)

    for arg in args:
        expr_storage_D_arg_storage = expr_storage.jacobian(ops.StorageOps.to_storage(arg))
        arg_jacobian = (
            expr_tangent_D_storage
            * expr_storage_D_arg_storage
            * ops.LieGroupOps.storage_D_tangent(arg)
        )

        jacobians.append(arg_jacobian)

    return jacobians
