# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import dataclasses

from symforce import typing as T
from symforce import typing_util
from symforce.ops import StorageOps


class DataclassStorageOps:
    """
    StorageOps implementation for dataclasses

    Supports nested types.  If any of the fields are of unknown size (e.g. sequences), the relevant
    functions expect to be passed an instance instead of the type. However, the length of sequences
    can be specified using field metadata, allowing for StorageOps functions such as "storage_dim",
    "from_storage", and "symbolic" to be passed the dataclass type rather than an instance. Adding
    a sequence of length 10, for example, would look like:

    @dataclass
    class ExampleDataclass:
        example_list: T.Sequence[ExampleType] = field(metadata={"length": 10})


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
                field_type = type_hints_map[field.name]
                if field.metadata.get("length") is not None:
                    sequence_instance = typing_util.get_sequence_from_dataclass_sequence_field(
                        field, field_type
                    )
                    count += StorageOps.storage_dim(sequence_instance)
                elif (
                    sequence_types := typing_util.maybe_tuples_of_types_from_annotation(field_type)
                ) is not None:
                    # It's a Tuple of known size
                    count += StorageOps.storage_dim(sequence_types)
                else:
                    count += StorageOps.storage_dim(field_type)
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
                if field.metadata.get("length") is not None:
                    sequence_instance = typing_util.get_sequence_from_dataclass_sequence_field(
                        field, field_type
                    )
                    storage_dim = StorageOps.storage_dim(sequence_instance)
                    constructed_fields[field.name] = StorageOps.from_storage(
                        sequence_instance, elements[offset : offset + storage_dim]
                    )
                elif (
                    sequence_types := typing_util.maybe_tuples_of_types_from_annotation(field_type)
                ) is not None:
                    # It's a Tuple of known size
                    storage_dim = StorageOps.storage_dim(sequence_types)
                    constructed_fields[field.name] = StorageOps.from_storage(
                        sequence_types, elements[offset : offset + storage_dim]
                    )
                else:
                    storage_dim = StorageOps.storage_dim(field_type)
                    constructed_fields[field.name] = StorageOps.from_storage(
                        field_type, elements[offset : offset + storage_dim]
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
                    field_instance, elements[offset : offset + storage_dim]
                )
                offset += storage_dim
            return typing_util.get_type(a)(**constructed_fields)

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
                    if field.metadata.get("length") is not None:
                        sequence_instance = typing_util.get_sequence_from_dataclass_sequence_field(
                            field, field_type
                        )
                        constructed_fields[field.name] = StorageOps.symbolic(
                            sequence_instance, f"{name_prefix}{field.name}", **kwargs
                        )
                    elif (
                        sequence_types := typing_util.maybe_tuples_of_types_from_annotation(
                            field_type
                        )
                    ) is not None:
                        # It's a Tuple of known size
                        constructed_fields[field.name] = StorageOps.symbolic(
                            sequence_types, f"{name_prefix}{field.name}", **kwargs
                        )
                    else:
                        constructed_fields[field.name] = StorageOps.symbolic(
                            field_type, f"{name_prefix}{field.name}", **kwargs
                        )
                except NotImplementedError as ex:
                    raise NotImplementedError(
                        f"Could not create field {field.name} of type {field_type}"
                    ) from ex
            return typing_util.get_type(a)(**constructed_fields)
        else:
            constructed_fields = {}
            name_prefix = f"{name}." if name is not None else ""
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
            return typing_util.get_type(a)(**constructed_fields)
