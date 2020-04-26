"""
Base classes for symengine geometric types.

These types implement methods that fulfill the concepts defined in `symforce.ops`.
"""

from symforce import ops
from symforce import sympy as sm


class Storage(object):
    # Copy docstring from concept class
    __doc__ = ops.StorageOps.__doc__

    # Dimension of underlying storage
    STORAGE_DIM = None

    def __repr__(self):
        """
        String representation of this type.

        Returns:
            str:
        """
        raise NotImplementedError()

    def to_storage(self):
        """
        Flat list representation of the underlying storage.
        This is used purely for plumbing, it is NOT like a tangent space.

        Returns:
            list: length of STORAGE_DIM
        """
        raise NotImplementedError()

    @classmethod
    def from_storage(cls, elements):
        """
        Construct from a flat list representation. Opposite of `.to_storage()`.

        Args:
            elements (list):

        Returns:
            Storage:
        """
        raise NotImplementedError()

    def __eq__(self, other):
        """
        Returns exact equality between self and other.

        Args:
            other (Storage):

        Returns:
            bool:
        """
        if not isinstance(self, other.__class__):
            return False

        self_list, other_list = self.to_storage(), other.to_storage()
        if not len(self_list) == len(other_list):
            return False

        if not all(s == o for s, o in zip(self_list, other_list)):
            return False

        return True

    def subs(self, *args, **kwargs):
        """
        Substitute given given values of each scalar element into a new instance.

        Returns:
            Storage: Substituted expression
        """
        return self.from_storage(sm.Matrix(self.to_storage()).subs(*args, **kwargs))

    # TODO(hayk): Way to get sm.simplify to work on these types directly?
    def simplify(self):
        """
        Simplify each scalar element into a new instance.

        Returns:
            Storage: Simplified expression
        """
        return self.from_storage(sm.simplify(sm.Matrix(self.to_storage())))

    @classmethod
    def symbolic(cls, name, **kwargs):
        """
        Construct a symbolic element with the given name prefix.

        Args:
            name (str): String prefix
            kwargs (dict): Additional arguments to pass to sm.Symbol (like assumptions)

        Returns:
            Storage:
        """
        return cls.from_storage(
            [sm.Symbol("{}_{}".format(name, i), **kwargs) for i in range(cls.STORAGE_DIM)]
        )

    def evalf(self):
        """
        Numerical evaluation.

        Returns:
            Storage:
        """
        return self.from_storage([ops.StorageOps.evalf(e) for e in self.to_storage()])


class Group(Storage):
    # Copy docstring from concept class
    __doc__ = ops.GroupOps.__doc__

    @classmethod
    def identity(cls):
        """
        Identity element such that `compose(a, identity) = a`.

        Returns:
            Group:
        """
        raise NotImplementedError()

    def compose(self, other):
        """
        Apply the group operation with other.

        Args:
            other (Group):

        Returns:
            Group:
        """
        raise NotImplementedError()

    def inverse(self):
        """
        Group inverse, such that `compose(a, inverse(a)) = a`.

        Returns:
            Group:
        """
        raise NotImplementedError()

    def between(self, b):
        """
        Returns the element that when composed with this produces b. For vector spaces it is `this - a`.

        Implementation is simply `compose(inverse(this), b)`.

        Args:
            b (Group):

        Returns:
            Group:
        """
        return self.inverse().compose(b)


class LieGroup(Group):
    # Copy docstring from concept class
    __doc__ = ops.LieGroupOps.__doc__

    # Dimension of the embedded manifold
    TANGENT_DIM = None

    # Dimensions of the Euclidean space that contains the manifold
    MATRIX_DIMS = None

    @classmethod
    def from_tangent(cls, vec, epsilon=0):
        """
        Mapping from the tangent space vector about identity into a group element.

        Args:
            vec (list):
            epsilon (Scalar): Small number to avoid singularity

        Returns:
            LieGroup:
        """
        if hasattr(cls, "expmap"):
            return cls.expmap(vec, epsilon=epsilon)

        raise NotImplementedError()

    def to_tangent(self, epsilon=0):
        """
        Mapping from this element to the tangent space vector about identity.

        Args:
            epsilon (Scalar): Small number to avoid singularity

        Returns:
            list:
        """
        if hasattr(self, "logmap"):
            return self.logmap(epsilon=epsilon)

        raise NotImplementedError()

    def retract(self, vec, epsilon=0):
        """
        Apply a tangent space pertubation vec to this. Often used in optimization
        to update nonlinear values from an update step in the tangent space.

        Implementation is simply `compose(this, from_tangent(vec))`.

        Args:
            vec (list):
            epsilon (Scalar): Small number to avoid singularity

        Returns:
            LieGroup: Group element that conceptually represents "this + vec"
        """
        return self.compose(self.from_tangent(vec, epsilon=epsilon))

    def local_coordinates(self, b, epsilon=0):
        """
        Computes a tangent space pertubation around this to produce b. Often used in optimization
        to minimize the distance between two group elements.

        Implementation is simply `to_tangent(between(this, b))`.

        Args:
            b (LieGroup):
            epsilon (Scalar): Small number to avoid singularity

        Returns:
            list: Tangent space pertubation that conceptually represents "this - a"
        """
        return self.between(b).to_tangent(epsilon=epsilon)
