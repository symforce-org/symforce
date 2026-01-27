# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from enum import Enum


class DType(Enum):
    FLOAT = "float"
    DOUBLE = "double"

    def is_double(self) -> bool:
        return self == DType.DOUBLE

    def is_float(self) -> bool:
        return self == DType.FLOAT

    def byte_size(self) -> int:
        if self == DType.FLOAT:
            return 4
        elif self == DType.DOUBLE:
            return 8

    def lower(self) -> str:
        return self.value.lower()

    def capitalize(self) -> str:
        return self.value.capitalize()

    def __str__(self) -> str:
        return self.value
