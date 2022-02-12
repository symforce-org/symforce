# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import inspect

from symforce import typing as T


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

    # An item of IMPLEMENTATIONS is of the form (DataType, (OpsClass, Implementation)),
    # where OpsClass is the subclass of Ops we are registering with, DataType is the
    # type we are registering with OpsClass, and Implementation is the class whose
    # methods know how to perform OpsClass operations on DataType.
    #
    # For example, the item that is added by StorageOps.register(float, ScalarStorageOps)
    # is (float, (StorageOps, ScalarStorageOps)).
    IMPLEMENTATIONS: T.Dict[T.Type, T.Tuple[T.Type, T.Type]] = {}

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
        cls.IMPLEMENTATIONS[impl_type] = (cls, impl_ops)

    @classmethod
    def implementation(cls, impl_type: T.Type) -> T.Type:
        """
        Returns the class defining the operations for the given type or one of
        its parent classes. If multiple parent classes are registered with the calling class,
        the implementation of the first such parent class in method resolution order is returned.

        Raises:
            NotImplementedError: If impl_type or one of its parent classes is not registered
            with the calling class or one of its subclasses.
        """
        registered_and_base: T.List[T.Tuple[T.Type, T.Type]] = []
        for base_class in inspect.getmro(impl_type):
            reg_class_and_impl = cls.IMPLEMENTATIONS.get(base_class, None)
            if reg_class_and_impl is not None:
                registration_class, impl = reg_class_and_impl
                if issubclass(registration_class, cls):
                    return impl

                registered_and_base.append((registration_class, base_class))

        # Handle Protocols/Metaclasses that aren't in the MRO (method resolution order), by trying
        # every class that's registered, and checking if impl_type is a subclass.
        for base_class, (registration_class, impl) in cls.IMPLEMENTATIONS.items():
            if issubclass(impl_type, base_class):
                if issubclass(registration_class, cls):
                    return impl

                # TODO(aaron): This will double count things with the above loop
                registered_and_base.append((registration_class, base_class))

        if len(registered_and_base) != 0:
            raise NotImplementedError(
                f"While {impl_type} is registered under {{reg_info}}, {{none_do}} subclass {cls.__name__}.".format(
                    reg_info=", ".join(
                        [
                            f"{reg.__name__} (via base class {base})"
                            for reg, base in registered_and_base
                        ]
                    ),
                    none_do="that does not" if len(registered_and_base) == 1 else "none of those",
                )
            )

        raise NotImplementedError(f"{impl_type} is not registered under {cls.__name__}")
