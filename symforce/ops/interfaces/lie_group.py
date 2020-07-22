from symforce import ops
from symforce import types as T

from .group import Group


class LieGroup(Group):
    """
    Interface for objects that implement the lie group concept. Because this
    class is registered using ClassLieGroupOps (see bottom of this file), any
    object that inherits from LieGroup and that implements the functions defined
    in this class can be used with the LieGroupOps concept.

    Note that LieGroup is a subclass of Group which is a subclass of Storage,
    meaning that a LieGroup object can be also be used with GroupOps and
    StorageOps (assuming the necessary functions are implemented).
    """

    # Type that represents this or any subclasses
    LieGroupT = T.TypeVar("LieGroupT", bound="LieGroup")

    @classmethod
    def tangent_dim(cls):
        # type: () -> int
        """
        Dimension of the embedded manifold
        """
        raise NotImplementedError()

    @classmethod
    def from_tangent(
        cls,  # type: T.Type[LieGroupT]
        vec,  # type: T.Sequence[T.Scalar]
        epsilon=0,  # type: T.Scalar
    ):
        # type: (...) -> LieGroupT
        """
        Mapping from the tangent space vector about identity into a group element.
        """
        raise NotImplementedError()

    def to_tangent(
        self,  # type: LieGroupT
        epsilon=0,  # type: T.Scalar
    ):
        # type: (...) -> T.List[T.Scalar]
        """
        Mapping from this element to the tangent space vector about identity.
        """
        raise NotImplementedError()

    def retract(
        self,  # type: LieGroupT
        vec,  # type: T.Sequence[T.Scalar]
        epsilon=0,  # type: T.Scalar
    ):
        # type: (...) -> LieGroupT
        """
        Apply a tangent space pertubation vec to this. Often used in optimization
        to update nonlinear values from an update step in the tangent space.

        Implementation is simply `compose(this, from_tangent(vec))`.
        Conceptually represents "this + vec".
        """
        return self.compose(self.from_tangent(vec, epsilon=epsilon))

    def local_coordinates(
        self,  # type: LieGroupT
        b,  # type: LieGroupT
        epsilon=0,  # type: T.Scalar
    ):
        # type: (...) -> T.List[T.Scalar]
        """
        Computes a tangent space pertubation around this to produce b. Often used in optimization
        to minimize the distance between two group elements.

        Implementation is simply `to_tangent(between(this, b))`.
        Tangent space pertubation that conceptually represents "this - a".
        """
        return self.between(b).to_tangent(epsilon=epsilon)


from ..impl.class_lie_group_ops import ClassLieGroupOps

ops.LieGroupOps.register(LieGroup, ClassLieGroupOps)
