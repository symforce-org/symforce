# This file automatically generated by skymarshal
# DO NOT MODIFY BY HAND
# fmt: off
# isort: off
# mypy: disallow-untyped-defs

import copy
import typing as T  # pylint: disable=unused-import

from io import BytesIO
import struct
from lcmtypes.eigen_lcm._Vector4d import Vector4d

class my_dataclass_t(object):
    __slots__: T.List[str] = ["rot"]

    def __init__(
        self,
        rot: T.Optional[Vector4d]=None,
        _skip_initialize: bool=False,
    ) -> None:
        """ If _skip_initialize is True, all other constructor arguments are ignored """
        if _skip_initialize:
            return
        self.rot: Vector4d = Vector4d._default() if rot is None else rot

    @staticmethod
    def from_all_fields(
        rot: Vector4d,
    ) -> "my_dataclass_t":
        return my_dataclass_t(
            rot=rot,
        )

    @staticmethod
    def _skytype_meta() -> T.Dict[str, str]:
        return dict(
            type="struct",
            package="codegen_test",
            name="my_dataclass_t",
        )

    @classmethod
    def _default(cls) -> "my_dataclass_t":
        return cls()

    def __repr__(self) -> str:
        return "my_dataclass_t({})".format(
            ", ".join("{}={}".format(name, repr(getattr(self, name))) for name in self.__slots__))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, my_dataclass_t):
            return NotImplemented
        return (
            (self.rot==other.rot)
        )
    # Disallow hashing for python struct lcmtypes.
    __hash__ = None  # type: ignore[assignment]

    def encode(self) -> bytes:
        buf = BytesIO()
        buf.write(my_dataclass_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf: T.BinaryIO) -> None:
        if hasattr(self.rot, '_get_packed_fingerprint'):
            assert self.rot._get_packed_fingerprint() == Vector4d._get_packed_fingerprint()
        else:
            assert self.rot._get_hash_recursive([]) == Vector4d._get_hash_recursive([])
        self.rot._encode_one(buf)

    @staticmethod
    def decode(data: T.Union[bytes, T.BinaryIO]) -> "my_dataclass_t":
        # NOTE(eric): This function can technically accept either a BinaryIO or
        # anything that supports the C++ Buffer Protocol,
        # which is unspecifiable in type hints.

        if hasattr(data, "read"):
            # NOTE(eric): mypy isn't able to figure out the hasattr check
            buf = T.cast(T.BinaryIO, data)
        else:
            buf = BytesIO(T.cast(bytes, data))

        if buf.read(8) != my_dataclass_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return my_dataclass_t._decode_one(buf)

    @staticmethod
    def _decode_one(buf: T.BinaryIO) -> "my_dataclass_t":
        self = my_dataclass_t(_skip_initialize=True)
        self.rot = Vector4d._decode_one(buf)
        return self

    @staticmethod
    def _get_hash_recursive(parents: T.List[T.Type]) -> int:
        if my_dataclass_t in parents: return 0
        newparents = parents + [my_dataclass_t]
        tmphash = (0x34567803726f7424+ Vector4d._get_hash_recursive(newparents)) & 0xffffffffffffffff
        tmphash = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash

    _packed_fingerprint: T.Optional[bytes] = None

    @staticmethod
    def _get_packed_fingerprint() -> bytes:
        if my_dataclass_t._packed_fingerprint is None:
            my_dataclass_t._packed_fingerprint = struct.pack(">Q", my_dataclass_t._get_hash_recursive([]))
        return my_dataclass_t._packed_fingerprint

    def deepcopy(self, **kwargs: T.Any) -> "my_dataclass_t":
        """
        Deep copy of this LCM type

        Returns a copy w/ members specified by kwargs replaced with new values specified by kwargs.
        """
        result = copy.deepcopy(self)
        for key in kwargs:
            if not hasattr(result, key):
                raise KeyError("Type my_dataclass_t does not have attribute: " + str(key))
            setattr(result, key, kwargs[key])
        return result
