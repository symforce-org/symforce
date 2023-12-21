# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import sym
import symforce.symbolic as sf
from symforce import codegen
from symforce import typing as T
from symforce.codegen.similarity_index import SimilarityIndex
from symforce.opt._internal.generated_residual_cache import GeneratedResidualCache
from symforce.test_util import TestCase
from symforce.values import Values


class GeneratedResidualCacheTest(TestCase):
    """
    Tests symforce.opt.generated_residual_cache.GeneratedResidualCache.
    """

    @staticmethod
    def example_index_keys_and_residual() -> (
        T.Tuple[SimilarityIndex, T.Set[str], T.Dict[str, T.Any], T.Callable]
    ):
        index = SimilarityIndex(
            inputs=Values(rot=sf.Rot3.symbolic("a")),
            outputs=Values(out=sf.Rot3.symbolic("a").inverse()),
            config=codegen.PythonConfig(),
            return_key="out",
            sparse_matrices=[],
        )

        optimized_keys = {"rot"}

        extra_args = dict(output_dir=None, namespace=None, sparse_linearization=False)

        def residual(a: sym.Rot3) -> sym.Rot3:
            return a.inverse()

        return index, optimized_keys, extra_args, residual

    def test_residual_can_be_retrieved(self) -> None:
        """
        Tests:
            GeneratedResidualCache.cache_residual
            GeneratedResidualCache.get_residual

        Simply that residuals stored with cache_residual can be retrieved with get_residual.
        """

        index, optimized_keys, extra_args, residual = self.example_index_keys_and_residual()

        cache = GeneratedResidualCache()

        with self.subTest(msg="Returns None if not cached"):
            self.assertEqual(
                cache.get_residual(index=index, optimized_keys=optimized_keys, **extra_args),
                None,
            )

        cache.cache_residual(
            index=index, optimized_keys=optimized_keys, **extra_args, residual=residual
        )

        with self.subTest(msg="Returns cached residual if cached"):
            self.assertEqual(
                cache.get_residual(index=index, optimized_keys=optimized_keys, **extra_args),
                residual,
            )

    def test_mutating_keys_does_not_break_cache(self) -> None:
        """
        Tests:
            GeneratedResidualCache.cache_residual
            GeneratedResidualCache.get_residual

        Specifically, if values passed into cache_residual are mutated after the function
        call, then cache_residual still works as intended.
        """

        index, optimized_keys, extra_args, residual = self.example_index_keys_and_residual()

        cache = GeneratedResidualCache()
        cache.cache_residual(
            index=index, optimized_keys=optimized_keys, residual=residual, **extra_args
        )

        index.outputs["out2"] = index.outputs["out"].inverse()
        optimized_keys.clear()

        og_index, og_optimized_keys, extra_args, _ = self.example_index_keys_and_residual()

        with self.subTest(msg="Cached key is not modified if argument index is mutated"):
            self.assertEqual(
                cache.get_residual(index=index, optimized_keys=og_optimized_keys, **extra_args),
                None,
            )

        with self.subTest(msg="Cached key is not modified if argument keys are mutated"):
            self.assertEqual(
                cache.get_residual(index=og_index, optimized_keys=optimized_keys, **extra_args),
                None,
            )

        with self.subTest(msg="Original cached value is still retrievable with original arguments"):
            self.assertEqual(
                cache.get_residual(index=og_index, optimized_keys=og_optimized_keys, **extra_args),
                residual,
            )


if __name__ == "__main__":
    TestCase.main()
