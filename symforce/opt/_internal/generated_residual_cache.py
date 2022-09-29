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
    output_dir: T.Optional[T.Openable]
    namespace: T.Optional[str]
    sparse_linearization: bool


class GeneratedResidualCache:
    """
    Utility class for keeping track of residuals which have already been generated.

    Functions like a dict whose keys are pairs of codegen_util.SimilarityIndex
    and T.Set[str], and whose values are residuals.

    Since the "keys" are made up of mutable objects, they are deep copied when cached.

    Not all of the components of the key here affect the stored residual, so there may be multiple
    copies of the same residual function in this cache.  What they do indicate is whether the
    Factor class that uses these cache needs to take other actions that have side effects, e.g.
    generate the same residual into a different output directory.
    """

    def __init__(self) -> None:
        self._dict: T.Dict[_GRCKey, T.Callable] = {}

    def get_residual(
        self,
        index: SimilarityIndex,
        optimized_keys: T.Iterable[str],
        output_dir: T.Optional[T.Openable],
        namespace: T.Optional[str],
        sparse_linearization: bool,
    ) -> T.Optional[T.Callable]:
        """
        If a residual function has already been cached using cache_residual with
        the given arguments, returns it.

        Otherwise, returns None.
        """

        return self._dict.get(
            _GRCKey(
                index=index,
                optimized_keys=tuple(sorted(optimized_keys)),
                output_dir=output_dir,
                namespace=namespace,
                sparse_linearization=sparse_linearization,
            ),
            None,
        )

    def cache_residual(
        self,
        index: SimilarityIndex,
        optimized_keys: T.Iterable[str],
        output_dir: T.Optional[T.Openable],
        namespace: T.Optional[str],
        sparse_linearization: bool,
        residual: T.Callable,
    ) -> None:
        """
        Caches residual so that it can be retrieved with get_residual(*key_args).

        Performs a deepcopy of any mutable arguments.
        """

        self._dict[
            _GRCKey(
                index=copy.deepcopy(index),
                optimized_keys=tuple(sorted(optimized_keys)),
                output_dir=output_dir,
                namespace=namespace,
                sparse_linearization=sparse_linearization,
            )
        ] = residual

    def __len__(self) -> int:
        """
        Returns the number of entries in the cache
        """
        return len(self._dict)
