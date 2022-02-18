# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import dataclasses

from symforce.ops import StorageOps
from symforce.python_util import get_type
from symforce import typing as T


class DataclassStorageOps:
    """
    StorageOps implementation for dataclasses

    Supports nested types.  If any of the fields are of unknown size (e.g. sequences), the relevant
    functions expect to be passed an instance instead of the type.

    NOTE(aaron): We use T.get_type_hints in multiple places in here to the field types, does this
    always work?  A bit worried that this never uses field.type, e.g. if it isn't a simple
    annotation
    """

    @staticmethod
    def storage_dim(a: T.DataclassOrType) -> int:
        if isinstance(a, type):
            count = 0
            type_hints_map = T.get_type_hints(a)
            for field in dataclasses.fields(a):
                count += StorageOps.storage_dim(type_hints_map[field.name])
            return count
        else:
            count = 0
            for field in dataclasses.fields(a):
                count += StorageOps.storage_dim(getattr(a, field.name))
            return count

    @staticmethod
    def to_storage(a: T.Dataclass) -> T.List[T.Scalar]:
        storage = []
        for field in dataclasses.fields(a):
            storage.extend(StorageOps.to_storage(getattr(a, field.name)))
        return storage

    @staticmethod
    def from_storage(a: T.DataclassOrType, elements: T.Sequence[T.Scalar]) -> T.Dataclass:
        if isinstance(a, type):
            constructed_fields = {}
            offset = 0
            type_hints_map = T.get_type_hints(a)
            for field in dataclasses.fields(a):
                field_type = type_hints_map[field.name]
                storage_dim = StorageOps.storage_dim(field_type)
                constructed_fields[field.name] = StorageOps.from_storage(
                    field_type, elements[offset : offset + storage_dim],
                )
                offset += storage_dim
            return a(**constructed_fields)
        else:
            constructed_fields = {}
            offset = 0
            for field in dataclasses.fields(a):
                field_instance = getattr(a, field.name)
                storage_dim = StorageOps.storage_dim(field_instance)
                constructed_fields[field.name] = StorageOps.from_storage(
                    field_instance, elements[offset : offset + storage_dim],
                )
                offset += storage_dim
            return get_type(a)(**constructed_fields)

    @staticmethod
    def symbolic(a: T.DataclassOrType, name: T.Optional[str], **kwargs: T.Dict) -> T.Dataclass:
        """
        Return a symbolic instance of a Dataclass

        Names are chosen by creating each field with symbolic name {name}.{field_name}.  If the
        `name` argument is not given, that part is left off, and fields are created with just
        {field_name}.
        """
        if isinstance(a, type):
            constructed_fields = {}

            name_prefix = f"{name}." if name is not None else ""
            type_hints_map = T.get_type_hints(a)
            for field in dataclasses.fields(a):
                field_type = type_hints_map[field.name]

                try:
                    constructed_fields[field.name] = StorageOps.symbolic(
                        field_type, f"{name_prefix}{field.name}", **kwargs
                    )
                except NotImplementedError as ex:
                    raise NotImplementedError(
                        f"Could not create field {field.name} of type {field_type}"
                    ) from ex
            return get_type(a)(**constructed_fields)
        else:
            constructed_fields = {}

            name_prefix = f"{name}." if name is not None else ""
            type_hints_map = T.get_type_hints(type(a))
            for field in dataclasses.fields(a):
                field_instance = getattr(a, field.name)

                try:
                    constructed_fields[field.name] = StorageOps.symbolic(
                        field_instance, f"{name_prefix}{field.name}", **kwargs
                    )
                except NotImplementedError as ex:
                    raise NotImplementedError(
                        f"Could not create field {field.name} of type {field_instance}"
                    ) from ex
            return get_type(a)(**constructed_fields)
