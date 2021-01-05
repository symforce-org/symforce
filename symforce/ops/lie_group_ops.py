from symforce import types as T
from symforce import sympy as sm
from symforce import logger
from symforce.python_util import get_type

from .ops import Ops
from .storage_ops import StorageOps
from .group_ops import GroupOps

Element = T.Any
ElementOrType = T.Union[Element, T.Type]

if T.TYPE_CHECKING:
    from symforce import geo


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
    def tangent_dim(a: ElementOrType) -> int:
        """
        Size of the element's tangent space, aka the degrees of freedom it represents. The
        storage_dim is the higher dimensional space in which this manifold is embedded. For
        example SO3 could be a tangent_dim of 3 with a storage_dim of 4 if storing quaternions,
        or 9 if storing rotation matrices. For vector spaces they are equal.
        """
        return Ops.implementation(get_type(a)).tangent_dim(a)

    @staticmethod
    def from_tangent(a: ElementOrType, vec: T.List[T.Scalar], epsilon: T.Scalar = 0) -> Element:
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
        return Ops.implementation(get_type(a)).from_tangent(a, vec, epsilon)

    @staticmethod
    def to_tangent(a: Element, epsilon: T.Scalar = 0) -> T.List[T.Scalar]:
        """
        Mapping from this element to the tangent space approximation at identity.

        Args:
            a (Element):
            epsilon (Scalar): Small number to avoid singularity

        Returns:
            list: Tangent space pertubation around identity that approximates a.
        """
        return Ops.implementation(get_type(a)).to_tangent(a, epsilon)

    @staticmethod
    def retract(a: Element, vec: T.List[T.Scalar], epsilon: T.Scalar = 0) -> Element:
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
        return LieGroupOps.compose(a, LieGroupOps.from_tangent(a, vec, epsilon=epsilon))

    @staticmethod
    def local_coordinates(a: Element, b: Element, epsilon: T.Scalar = 0) -> T.List[T.Scalar]:
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
        return LieGroupOps.to_tangent(LieGroupOps.between(a, b), epsilon=epsilon)

    @staticmethod
    def storage_D_tangent(a: Element) -> "geo.Matrix":
        """
        Computes the jacobian of the storage space of an element with respect to the tangent space around
        that element.
        """
        try:
            return Ops.implementation(get_type(a)).storage_D_tangent(a)
        except NotImplementedError:
            logger.error(
                "storage_D_tangent not implemented for {}; use storage_D_tangent.ipynb to compute".format(
                    get_type(a)
                )
            )
            raise NotImplementedError()

    @staticmethod
    def tangent_D_storage(a: Element, epsilon: Element = 0) -> "geo.Matrix":
        """
        Computes the jacobian of the tangent space around an element with respect to the storage space of
        that element.
        """
        try:
            return Ops.implementation(get_type(a)).tangent_D_storage(a)
        except NotImplementedError:
            logger.error(
                "tangent_D_storage not implemented for {}; use tangent_D_storage.ipynb to compute".format(
                    get_type(a)
                )
            )
            raise NotImplementedError()
