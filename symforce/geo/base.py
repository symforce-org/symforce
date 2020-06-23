"""
Base classes for symengine geometric types.

These types implement methods that fulfill the concepts defined in `symforce.ops`.
"""

from symforce import ops
from symforce import sympy as sm
from symforce import types as T


class Storage(object):
    # Copy docstring from concept class
    __doc__ = ops.StorageOps.__doc__

    # Dimension of underlying storage
    # TODO(nathan, hayk): Change STORAGE_DIM to classmethod storage_dim() to match C++ and avoid
    # avoid complexities with inheritence (i.e. classproperty)
    STORAGE_DIM = -1  # type: int

    # Type that represents this or any subclasses
    StorageT = T.TypeVar("StorageT", bound="Storage")

    def __repr__(self):
        # type: (StorageT) -> str
        """
        String representation of this type.
        """
        raise NotImplementedError()

    def to_storage(self):
        # type: (StorageT) -> T.List[T.Scalar]
        """
        Flat list representation of the underlying storage, length of STORAGE_DIM.
        This is used purely for plumbing, it is NOT like a tangent space.
        """
        raise NotImplementedError()

    @classmethod
    def from_storage(
        cls,  # type: T.Type[StorageT]
        elements,  # type: T.Sequence[T.Scalar]
    ):
        # type: (...) -> StorageT
        """
        Construct from a flat list representation. Opposite of `.to_storage()`.
        """
        raise NotImplementedError()

    def __eq__(
        self,  # type: StorageT
        other,  # type: T.Any
    ):
        # type: (...) -> bool
        """
        Returns exact equality between self and other.
        """
        if not isinstance(self, other.__class__):
            return False

        self_list, other_list = self.to_storage(), other.to_storage()
        if not len(self_list) == len(other_list):
            return False

        if not all(s == o for s, o in zip(self_list, other_list)):
            return False

        return True

    def subs(
        self,  # type: StorageT
        *args,  # type: T.Any
        **kwargs  # type: T.Any
    ):
        # type: (...) -> StorageT
        """
        Substitute given values of each scalar element into a new instance.
        """
        # TODO(hayk): If this is slow, compute the subs dict once.
        return self.from_storage([sm.S(s).subs(*args, **kwargs) for s in self.to_storage()])

    # TODO(hayk): Way to get sm.simplify to work on these types directly?
    def simplify(self):
        # type: (StorageT) -> StorageT
        """
        Simplify each scalar element into a new instance.
        """
        return self.from_storage(sm.simplify(sm.Matrix(self.to_storage())))

    @classmethod
    def symbolic(
        cls,  # type: T.Type[StorageT]
        name,  # type: str
        **kwargs  # type: T.Any
    ):
        # type: (...) -> StorageT
        """
        Construct a symbolic element with the given name prefix. Kwargs are forwarded
        to sm.Symbol (for example, sympy assumptions).
        """
        return cls.from_storage(
            [sm.Symbol("{}_{}".format(name, i), **kwargs) for i in range(cls.STORAGE_DIM)]
        )

    def evalf(self):
        # type: (StorageT) -> StorageT
        """
        Numerical evaluation.
        """
        return self.from_storage([ops.StorageOps.evalf(e) for e in self.to_storage()])

    def __hash__(self):
        # type: () -> int
        """
        Hash this object in immutable form, by combining all their scalar hashes.

        NOTE(hayk, nathan): This is somewhat dangerous because we don't always guarantee
        that Storage objects are immutable (e.g. geo.Matrix). If you add this object as
        a key to a dict, modify it, and access the dict, it will show up as another key
        because it breaks the abstraction that an object will maintain the same hash over
        its lifetime.
        """
        return tuple(self.to_storage()).__hash__()


class Group(Storage):
    # Copy docstring from concept class
    __doc__ = ops.GroupOps.__doc__

    # Type that represents this or any subclasses
    GroupT = T.TypeVar("GroupT", bound="Group")

    @classmethod
    def identity(cls):
        # type: (T.Type[GroupT]) -> GroupT
        """
        Identity element such that `compose(a, identity) = a`.
        """
        raise NotImplementedError()

    def compose(
        self,  # type: GroupT
        other,  # type: GroupT
    ):
        # type: (...) -> GroupT
        """
        Apply the group operation with other.
        """
        raise NotImplementedError()

    def inverse(self):
        # type: (GroupT) -> GroupT
        """
        Group inverse, such that `compose(a, inverse(a)) = a`.
        """
        raise NotImplementedError()

    def between(
        self,  # type: GroupT
        b,  # type: GroupT
    ):
        # type: (...) -> GroupT
        """
        Returns the element that when composed with this produces b. For vector spaces it is `this - a`.

        Implementation is simply `compose(inverse(this), b)`.
        """
        return self.inverse().compose(b)


class LieGroup(Group):
    # Copy docstring from concept class
    __doc__ = ops.LieGroupOps.__doc__

    # Dimension of the embedded manifold
    TANGENT_DIM = -1  # type: int

    # Type that represents this or any subclasses
    LieGroupT = T.TypeVar("LieGroupT", bound="LieGroup")

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
        if hasattr(cls, "expmap"):
            return cls.expmap(vec, epsilon=epsilon)  # type: ignore

        raise NotImplementedError()

    def to_tangent(
        self,  # type: LieGroupT
        epsilon=0,  # type: T.Scalar
    ):
        # type: (...) -> T.List[T.Scalar]
        """
        Mapping from this element to the tangent space vector about identity.
        """
        if hasattr(self, "logmap"):
            return self.logmap(epsilon=epsilon)  # type: ignore

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
