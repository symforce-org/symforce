import numpy as np
from symforce import sympy as sm


class StorageOps(object):
    """
    API for symbolic data types that can be serialized to and from a vector of scalar quantities.
    """

    @staticmethod
    def storage_dim(a):
        """
        Size of the element's storage, aka the number of scalar values it contains.

        Args:
            a (Element or type):

        Returns:
            int:
        """
        if hasattr(a, "STORAGE_DIM"):
            return a.STORAGE_DIM
        elif StorageOps.scalar_like(a):
            return 1
        elif StorageOps.array_like(a):
            return len(a)
        else:
            StorageOps._type_error(a)

    @staticmethod
    def to_storage(a):
        """
        Serialization of the underlying storage into a list. This is NOT a tangent space.

        Args:
            a (Element):
        Returns:
            list: Length equal to `storage_dim(a)`
        """
        if hasattr(a, "to_storage"):
            return a.to_storage()
        elif isinstance(a, (list, tuple, np.ndarray)):
            return list(a)
        elif StorageOps.scalar_like(a):
            return [a]
        else:
            StorageOps._type_error(a)

    @staticmethod
    def from_storage(cls, elements):
        """
        Construct from a flat list representation. Opposite of `.to_storage()`.

        Args:
            cls (type):
            elements (list):

        Returns:
            cls:
        """
        if hasattr(cls, "from_storage"):
            return cls.from_storage(elements)
        elif StorageOps.scalar_like(cls):
            assert len(elements) == 1, "Scalar needs one element."
            return elements[0]
        else:
            StorageOps._type_error(a)

    @staticmethod
    def _get_type(a):
        """
        Returns the type of the element if its an instance, or a pass through if already a type.

        Args:
            a (Element or type):

        Returns:
            type:
        """
        if isinstance(a, type):
            return a
        else:
            return type(a)

    @staticmethod
    def _type_error(a):
        """
        Raise an exception with type information.

        Args:
            a (Element or type):

        Raises:
            TypeError:
        """
        a_type = StorageOps._get_type(a)
        raise TypeError("val={}, type={}, mro={}, sm={}".format(a, a_type, a_type.__mro__, sm))

    @staticmethod
    def scalar_like(a):
        """
        Returns whether the element is scalar-like (an int, float, or sympy expression).

        This method does not rely on the value of a, only the type.

        Args:
            a (Element or type):

        Returns:
            bool:
        """
        a_type = StorageOps._get_type(a)
        if issubclass(a_type, (int, float, np.float32, np.float64)):
            return True
        is_expr = issubclass(a_type, sm.Expr)
        is_matrix = issubclass(a_type, sm.MatrixBase)
        return is_expr and not is_matrix

    @staticmethod
    def array_like(a):
        """
        Returns whether the element is array-like (tuple, list, numpy array).

        This method does not rely on the value of a, only the type.

        Args:
            a (Element or type):

        Returns:
            bool:
        """
        a_type = StorageOps._get_type(a)
        return issubclass(a_type, (list, tuple, np.ndarray))

    @staticmethod
    def evalf(a):
        """
        Evaluate to a numerical quantity (rationals, trig functions, etc).

        Args:
            a (Element):

        Returns:
            Element:
        """
        if hasattr(a, "evalf"):
            return a.evalf()
        elif StorageOps.scalar_like(a):
            return a
        elif StorageOps.array_like(a):
            return [StorageOps.evalf(v) for v in a]
        else:
            StorageOps._type_error(a)

    @staticmethod
    def symbolic(a, name, **kwargs):
        """
        Construct a symbolic element with the given name prefix.

        Args:
            a (Element or type):
            name (str): String prefix
            kwargs (dict): Additional arguments to pass to sm.Symbol (like assumptions)

        Returns:
            Storage:
        """
        if hasattr(a, "symbolic"):
            return a.symbolic(name, **kwargs)
        elif StorageOps.scalar_like(a):
            return sm.Symbol(name, **kwargs)
        elif StorageOps.array_like(a):
            return [
                sm.Symbol("{}_{}".format(name, i), **kwargs)
                for i in range(StorageOps.storage_dim(a))
            ]
        else:
            StorageOps._type_error(a)
