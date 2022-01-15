from symforce import typing as T

from .scalar_storage_ops import ScalarStorageOps


class ScalarGroupOps(ScalarStorageOps):
    @staticmethod
    def identity(_: T.ScalarElementOrType) -> T.ScalarElement:
        return 0.0

    @staticmethod
    def compose(a: T.ScalarElement, b: T.ScalarElement) -> T.ScalarElement:
        return a + b

    @staticmethod
    def inverse(a: T.ScalarElement) -> T.ScalarElement:
        return -a
