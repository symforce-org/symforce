import numpy as np
import inspect

from symforce import python_util
from symforce import sympy as sm
from symforce import types as T


class Ops:
    """
    Class for specifying how Storage/Group/LieGroup ops functions should
    be implemented for specific types (e.g. scalars, geo objects, etc.).
    Classes that inherit from Ops can be considered "concepts" (see
    https://en.wikipedia.org/wiki/Concept_(generic_programming) ),
    meaning that they define a set of valid operations on the types
    (or subtypes) registered with this base class.

    As classes are created, they (or one of their parent classes) must be
    registered by calling "register()", which specifies a specific implementation
    of the ops for that class. This is similar to template specialization in C++.
    """

    IMPLEMENTATIONS: T.Dict[T.Type, T.Type] = {}

    @classmethod
    def register(cls, impl_type: T.Type, impl_ops: T.Type) -> None:
        """
        Register the operations class for a given type. Once a type is
        registered, child classes of Ops will be able to call functions
        defined in impl_ops.

        Example:
            StorageOps.register(float, ScalarStorageOps)  # ScalarStorageOps defines valid storage operations on floats
            StorageOps.storage_dim(1.0)  # We can now perform storage operations of objects of type float

        Args:
            impl_type: Type to be registered
            impl_ops: Class defining how each operation is implemented for the given type
        """
        assert impl_type not in cls.IMPLEMENTATIONS
        cls.IMPLEMENTATIONS[impl_type] = impl_ops

    @classmethod
    def implementation(cls, impl_type: T.Type) -> T.Type:
        """
        Returns the class defining the operations for the given type or one of
        its parent classes.
        """
        for base_class in inspect.getmro(impl_type):
            impl = cls.IMPLEMENTATIONS.get(base_class, None)
            if impl is not None:
                return impl
        raise NotImplementedError(f"Unsupported type: {impl_type}")
