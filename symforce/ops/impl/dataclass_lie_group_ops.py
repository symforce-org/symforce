# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from __future__ import annotations

import dataclasses

from symforce import typing as T
from symforce import typing_util
from symforce.ops import LieGroupOps
from symforce.ops import StorageOps

from .dataclass_group_ops import DataclassGroupOps

if T.TYPE_CHECKING:
    from symforce import geo


class DataclassLieGroupOps(DataclassGroupOps):
    @staticmethod
    def tangent_dim(a: T.DataclassOrType) -> int:
        if isinstance(a, type):
            count = 0
            type_hints_map = T.get_type_hints(a)
            for field in dataclasses.fields(a):
                field_type = type_hints_map[field.name]
                if field.metadata.get("length") is not None:
                    sequence_instance = typing_util.get_sequence_from_dataclass_sequence_field(
                        field, field_type
                    )
                    count += LieGroupOps.tangent_dim(sequence_instance)
                elif (
                    sequence_types := typing_util.maybe_tuples_of_types_from_annotation(field_type)
                ) is not None:
                    # It's a Tuple of known size
                    count += LieGroupOps.tangent_dim(sequence_types)
                else:
                    count += LieGroupOps.tangent_dim(field_type)
            return count
        else:
            count = 0
            for field in dataclasses.fields(a):
                count += LieGroupOps.tangent_dim(getattr(a, field.name))
            return count

    @staticmethod
    def from_tangent(
        a: T.DataclassOrType, vec: T.Sequence[T.Scalar], epsilon: T.Scalar
    ) -> T.Dataclass:
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
                    tangent_dim = LieGroupOps.tangent_dim(sequence_instance)
                    constructed_fields[field.name] = LieGroupOps.from_tangent(
                        sequence_instance, vec[offset : offset + tangent_dim]
                    )
                elif (
                    sequence_types := typing_util.maybe_tuples_of_types_from_annotation(field_type)
                ) is not None:
                    # It's a Tuple of known size
                    tangent_dim = LieGroupOps.tangent_dim(sequence_types)
                    constructed_fields[field.name] = LieGroupOps.from_tangent(
                        sequence_types, vec[offset : offset + tangent_dim], epsilon
                    )
                else:
                    tangent_dim = LieGroupOps.tangent_dim(field_type)
                    constructed_fields[field.name] = LieGroupOps.from_tangent(
                        field_type, vec[offset : offset + tangent_dim], epsilon
                    )
                offset += tangent_dim
            return a(**constructed_fields)
        else:
            constructed_fields = {}
            offset = 0
            for field in dataclasses.fields(a):
                field_instance = getattr(a, field.name)
                tangent_dim = LieGroupOps.tangent_dim(field_instance)
                constructed_fields[field.name] = LieGroupOps.from_tangent(
                    field_instance, vec[offset : offset + tangent_dim], epsilon
                )
                offset += tangent_dim
            return typing_util.get_type(a)(**constructed_fields)

    @staticmethod
    def to_tangent(a: T.Dataclass, epsilon: T.Scalar) -> T.List[T.Scalar]:
        tangent = []
        for field in dataclasses.fields(a):
            tangent.extend(LieGroupOps.to_tangent(getattr(a, field.name), epsilon))
        return tangent

    @staticmethod
    def storage_D_tangent(a: T.Dataclass) -> geo.Matrix:
        from symforce import geo

        mat = geo.Matrix(StorageOps.storage_dim(a), LieGroupOps.tangent_dim(a))
        s_inx = 0
        t_inx = 0
        for field in dataclasses.fields(a):
            field_instance = getattr(a, field.name)
            s_dim = StorageOps.storage_dim(field_instance)
            t_dim = LieGroupOps.tangent_dim(field_instance)
            mat[s_inx : s_inx + s_dim, t_inx : t_inx + t_dim] = LieGroupOps.storage_D_tangent(
                field_instance
            )
            s_inx += s_dim
            t_inx += t_dim
        return mat

    @staticmethod
    def tangent_D_storage(a: T.Dataclass) -> geo.Matrix:
        from symforce import geo

        mat = geo.Matrix(LieGroupOps.tangent_dim(a), StorageOps.storage_dim(a))
        s_inx = 0
        t_inx = 0
        for field in dataclasses.fields(a):
            field_instance = getattr(a, field.name)
            s_dim = StorageOps.storage_dim(field_instance)
            t_dim = LieGroupOps.tangent_dim(field_instance)
            mat[t_inx : t_inx + t_dim, s_inx : s_inx + s_dim] = LieGroupOps.tangent_D_storage(
                field_instance
            )
            s_inx += s_dim
            t_inx += t_dim
        return mat

    @staticmethod
    def retract(a: T.Dataclass, vec: T.Sequence[T.Scalar], epsilon: T.Scalar) -> T.Dataclass:
        constructed_fields = {}
        offset = 0
        for field in dataclasses.fields(a):
            field_instance = getattr(a, field.name)
            tangent_dim = LieGroupOps.tangent_dim(field_instance)
            constructed_fields[field.name] = LieGroupOps.retract(
                field_instance, vec[offset : offset + tangent_dim], epsilon
            )
            offset += tangent_dim
        return typing_util.get_type(a)(**constructed_fields)

    @staticmethod
    def local_coordinates(a: T.Dataclass, b: T.Dataclass, epsilon: T.Scalar) -> T.List[T.Scalar]:
        assert typing_util.get_type(a) == typing_util.get_type(b)
        return [
            x
            for field in dataclasses.fields(a)
            for x in LieGroupOps.local_coordinates(
                getattr(a, field.name), getattr(b, field.name), epsilon
            )
        ]
