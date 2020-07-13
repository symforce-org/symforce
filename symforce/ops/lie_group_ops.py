from symforce import types as T
from symforce import sympy as sm
from symforce import logger

from .storage_ops import StorageOps
from .group_ops import GroupOps

Element = T.Any
ElementOrType = T.Union[Element, T.Type]


class LieGroupOps(GroupOps):
    """
    API for Lie groups.

    A Lie group is a group that is also a differentiable manifold, with the property that the
    group operations are compatible with the smooth structure.

    A manifold is a topological space that locally resembles Euclidean space near
    each point. More precisely, each point of an n-dimensional manifold has a neighbourhood that
    is homeomorphic to the Euclidean space of dimension n.

    A differentiable manifold is a type of manifold that is locally similar enough to a linear
    space to allow one to do calculus. Any manifold can be described by a collection of charts,
    also known as an atlas. One may then apply ideas from calculus while working within the
    individual charts, since each chart lies within a linear space to which the usual rules of
    calculus apply. If the charts are suitably compatible (namely, the transition from one chart
    to another is differentiable), then computations done in one chart are valid in any other
    differentiable chart.

    References:

        * https://en.wikipedia.org/wiki/Manifold
        * https://en.wikipedia.org/wiki/Differentiable_manifold
        * https://en.wikipedia.org/wiki/Lie_group
    """

    @staticmethod
    def tangent_dim(a):
        # type: (ElementOrType) -> int
        """
        Size of the element's tangent space, aka the degrees of freedom it represents. The
        storage_dim is the higher dimensional space in which this manifold is embedded. For
        example SO3 could be a tangent_dim of 3 with a storage_dim of 4 if storing quaternions,
        or 9 if storing rotation matrices. For vector spaces they are equal.
        """
        if hasattr(a, "TANGENT_DIM"):
            return a.TANGENT_DIM
        elif GroupOps.scalar_like(a):
            return 1
        else:
            GroupOps._type_error(a)

    @staticmethod
    def from_tangent(a, vec, epsilon=0):
        # type: (ElementOrType, T.List, T.Scalar) -> Element
        """
        Mapping from the tangent space approximation at identity into a group element of type a.
        For most manifold types this is implemented as the exponential map.

        This method does not rely on the value of a, only the type.

        Args:
            a (Element or type):
            vec (list): Tangent space pertubation
            epsilon (Scalar): Small number to avoid singularity

        Returns:
            Element: Valid group element that approximates vec around identity.
        """
        if hasattr(a, "from_tangent"):
            return a.from_tangent(vec, epsilon=epsilon)
        elif GroupOps.scalar_like(a):
            assert len(vec) == 1
            if isinstance(vec[0], sm.Symbol):
                return vec[0]
            constructor = a if isinstance(a, type) else type(a)
            return constructor(vec[0])
        else:
            GroupOps._type_error(a)

    @staticmethod
    def to_tangent(a, epsilon=0):
        # type: (Element, T.Scalar) -> T.List
        """
        Mapping from this element to the tangent space approximation at identity.

        Args:
            a (Element):
            epsilon (Scalar): Small number to avoid singularity

        Returns:
            list: Tangent space pertubation around identity that approximates a.
        """
        if hasattr(a, "to_tangent"):
            return a.to_tangent(epsilon=epsilon)
        elif GroupOps.scalar_like(a):
            return [a]
        else:
            GroupOps._type_error(a)

    @staticmethod
    def retract(a, vec, epsilon=0):
        # type: (Element, T.List, T.Scalar) -> Element
        """
        Apply a tangent space pertubation vec to the group element a. Often used in optimization
        to update nonlinear values from an update step in the tangent space.

        Implementation is simply `compose(a, from_tangent(vec))`.

        Args:
            a (Element):
            vec (list):
            epsilon (Scalar): Small number to avoid singularity

        Returns:
            Element: Group element that conceptually represents "a + vec"
        """
        if hasattr(a, "retract"):
            return a.retract(vec, epsilon=epsilon)

        return LieGroupOps.compose(a, LieGroupOps.from_tangent(a, vec, epsilon=epsilon))

    @staticmethod
    def local_coordinates(a, b, epsilon=0):
        # type: (Element, T.Any, T.Scalar) -> T.List
        """
        Computes a tangent space pertubation around a to produce b. Often used in optimization
        to minimize the distance between two group elements.

        Implementation is simply `to_tangent(between(a, b))`.

        Args:
            a (Element):
            b (Element):
            epsilon (Scalar): Small number to avoid singularity

        Returns:
            list: Tangent space pertubation that conceptually represents "b - a"
        """
        if hasattr(a, "local_coordinates"):
            return a.local_coordinates(b, epsilon=epsilon)

        return LieGroupOps.to_tangent(LieGroupOps.between(a, b), epsilon=epsilon)

    @staticmethod
    def storage_D_tangent(a, epsilon=0):
        # type: (Element) -> sm.Matrix
        """
        Computes the jacobian of the storage space of an element with respect to the tangent space around
        that element.
        """
        if hasattr(a, "storage_D_tangent"):
            # Use precomputed jacobian
            return a.storage_D_tangent()
        elif GroupOps.scalar_like(a):
            # TODO(nathan): Returning a sm.Matrix instead of a geo.Matrix could cause problems
            return sm.Matrix([1])
        else:
            a_type = a if isinstance(a, type) else type(a)
            logger.error(
                "storage_D_tangent not implemented for {}; use storage_D_tangent.ipynb to compute".format(
                    a_type
                )
            )
            GroupOps._type_error(a)
