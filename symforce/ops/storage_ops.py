import numpy as np
from symforce import sympy as sm
from symforce import types as T

Element = T.Any
ElementOrType = T.Union[Element, T.Type]


class StorageOps(object):
    """
    API for symbolic data types that can be serialized to and from a vector of scalar quantities.
    """

    @staticmethod
    def storage_dim(a):
        # type: (ElementOrType) -> int
        """
        Size of the element's storage, aka the number of scalar values it contains.
        """
        if hasattr(a, "STORAGE_DIM"):
            return a.STORAGE_DIM
        elif StorageOps.scalar_like(a):
            return 1
        elif StorageOps.array_like(a) and isinstance(a, T.Sized):
            return len(a)
        else:
            StorageOps._type_error(a)

    @staticmethod
    def to_storage(a):
        # type: (Element) -> T.List
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
        # type: (T.Type, T.List) -> T.Any
        """
        Construct from a flat list representation. Opposite of `.to_storage()`.
        """
        if hasattr(cls, "from_storage"):
            return cls.from_storage(elements)
        elif StorageOps.scalar_like(cls):
            assert len(elements) == 1, "Scalar needs one element."
            return elements[0]
        else:
            StorageOps._type_error(cls)

    @staticmethod
    def get_type(a):
        # type: (ElementOrType) -> T.Type
        """
        Returns the type of the element if its an instance, or a pass through if already a type.
        """
        if isinstance(a, type):
            return a
        else:
            return type(a)

    @staticmethod
    def _type_error(a):
        # type: (ElementOrType) -> T.NoReturn
        """
        Raise an exception with type information.
        """
        a_type = StorageOps.get_type(a)
        raise TypeError("val={}, type={}, mro={}, sm={}".format(a, a_type, a_type.__mro__, sm))

    @staticmethod
    def scalar_like(a):
        # type: (ElementOrType) -> bool
        """
        Returns whether the element is scalar-like (an int, float, or sympy expression).

        This method does not rely on the value of a, only the type.
        """
        a_type = StorageOps.get_type(a)
        if issubclass(a_type, (int, float, np.float32, np.float64)):
            return True
        is_expr = issubclass(a_type, sm.Expr)
        is_matrix = issubclass(a_type, sm.MatrixBase)
        return is_expr and not is_matrix

    @staticmethod
    def array_like(a):
        # type: (ElementOrType) -> bool
        """
        Returns whether the element is array-like (tuple, list, numpy array).

        This method does not rely on the value of a, only the type.
        """
        a_type = StorageOps.get_type(a)
        return issubclass(a_type, (list, tuple, np.ndarray))

    @staticmethod
    def evalf(a):
        # type: (Element) -> Element
        """
        Evaluate to a numerical quantity (rationals, trig functions, etc).
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
        # type: (ElementOrType, str, T.Dict) -> T.Any
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
