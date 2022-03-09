# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import copy
import dataclasses

from symforce import typing as T
from symforce.codegen.similarity_index import SimilarityIndex


@dataclasses.dataclass(frozen=True)
class _GRCKey:
    """
    Keys to the private dict of GeneratedResidualCache.
    """

    index: SimilarityIndex
    optimized_keys: T.Tuple[str, ...]


class GeneratedResidualCache:
    """
    Utility class for keeping track of residuals which have already been generated.

    Functions like a dict whose keys are pairs of codegen_util.SimilarityIndex
    and T.Set[str], and whose values are residuals.

    Since the "keys" are made up of mutable objects, they are deep copied when cached.
    """

    def __init__(self) -> None:
        self._dict: T.Dict[_GRCKey, T.Callable] = {}

    def get_residual(
        self, index: SimilarityIndex, optimized_keys: T.Iterable[str]
    ) -> T.Optional[T.Callable]:
        """
        If a residual function has already been cached using cache_residual with
        index and optimized keys, returns it.

        Otherwise, returns None.
        """

        return self._dict.get(
            _GRCKey(index=index, optimized_keys=tuple(sorted(optimized_keys))), None
        )

    def cache_residual(
        self, index: SimilarityIndex, optimized_keys: T.Iterable[str], residual: T.Callable
    ) -> None:
        """
        Caches residual so that it can be retrieved with get_residual(index, optmized_keys).

        Performs a deepcopy of index and optimized_keys.
        """

        self._dict[
            _GRCKey(index=copy.deepcopy(index), optimized_keys=tuple(sorted(optimized_keys)))
        ] = residual
