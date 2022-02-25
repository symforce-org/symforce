# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import dataclasses

from symforce.ops import GroupOps
from symforce.python_util import get_type, get_sequence_from_dataclass_sequence_field
from symforce import typing as T

from .dataclass_storage_ops import DataclassStorageOps


class DataclassGroupOps(DataclassStorageOps):
    @staticmethod
    def identity(a: T.DataclassOrType) -> T.Dataclass:
        constructed_fields = {}
        if isinstance(a, type):
            type_hints_map = T.get_type_hints(a)
            for field in dataclasses.fields(a):
                field_type = type_hints_map[field.name]
                if field.metadata.get("length") is not None:
                    sequence_instance = get_sequence_from_dataclass_sequence_field(
                        field, field_type
                    )
                    constructed_fields[field.name] = GroupOps.identity(sequence_instance)
                else:
                    constructed_fields[field.name] = GroupOps.identity(field_type)
            return a(**constructed_fields)
        else:
            for field in dataclasses.fields(a):
                constructed_fields[field.name] = GroupOps.identity(getattr(a, field.name))
            return get_type(a)(**constructed_fields)

    @staticmethod
    def compose(a: T.Dataclass, b: T.Dataclass) -> T.Dataclass:
        assert get_type(a) == get_type(b)
        constructed_fields = {}
        for field in dataclasses.fields(a):
            constructed_fields[field.name] = GroupOps.compose(
                getattr(a, field.name), getattr(b, field.name)
            )
        return get_type(a)(**constructed_fields)

    @staticmethod
    def inverse(a: T.Dataclass) -> T.Dataclass:
        constructed_fields = {}
        for field in dataclasses.fields(a):
            constructed_fields[field.name] = GroupOps.inverse(getattr(a, field.name))
        return get_type(a)(**constructed_fields)
