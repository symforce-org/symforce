# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce import codegen
from symforce.codegen.similarity_index import SimilarityIndex
from symforce.test_util import TestCase
from symforce.values import Values


class SymforceSimilarityIndexTest(TestCase):
    """
    Test symforce.codegen.similarity_index.SimilarityIndex
    """

    def test_from_codegen(self) -> None:
        """
        Tests:
            SimilarityIndex.from_codegen

        Test that the output of SimilarityIndex.from_codegen is equal to the index of another
        Codegen object if an only if the other codegen object would generate the same
        function (modulo differences in function name and doc_string).
        """

        config: codegen.CodegenConfig

        with self.subTest(msg="different function names don't imply dissimilarity"):
            config = codegen.PythonConfig()

            co_1 = codegen.Codegen.function(func=sf.Rot3.compose, name="some_name", config=config)
            co_2 = codegen.Codegen.function(
                func=sf.Rot3.compose, name="alternate_name", config=config
            )

            self.assertEqual(SimilarityIndex.from_codegen(co_1), SimilarityIndex.from_codegen(co_2))

        with self.subTest(msg="different doc-strings don't imply dissimilarity"):
            config = codegen.CppConfig()

            co_1 = codegen.Codegen.function(
                func=sf.Rot3.compose, docstring="a function", config=config
            )
            co_2 = codegen.Codegen.function(
                func=sf.Rot3.compose, docstring="multiply", config=config
            )

            self.assertEqual(SimilarityIndex.from_codegen(co_1), SimilarityIndex.from_codegen(co_2))

        with self.subTest(msg="Python functions are not similar to C++ functions"):
            co_cpp = codegen.Codegen.function(func=sf.Rot3.compose, config=codegen.CppConfig())
            co_python = codegen.Codegen.function(
                func=sf.Rot3.compose, config=codegen.PythonConfig()
            )

            self.assertNotEqual(
                SimilarityIndex.from_codegen(co_cpp), SimilarityIndex.from_codegen(co_python)
            )

        with self.subTest(msg="Different configuration settings are dissimilar"):
            co_no_numba = codegen.Codegen.function(
                func=sf.Rot3.compose, config=codegen.PythonConfig(use_numba=False)
            )
            co_with_numba = codegen.Codegen.function(
                func=sf.Rot3.compose, config=codegen.PythonConfig(use_numba=True)
            )

            self.assertNotEqual(
                SimilarityIndex.from_codegen(co_no_numba),
                SimilarityIndex.from_codegen(co_with_numba),
            )

        with self.subTest(msg="Different functions are dissimilar"):
            config = codegen.PythonConfig()

            co_rot3 = codegen.Codegen.function(func=sf.Rot3.compose, config=config)
            co_rot2 = codegen.Codegen.function(func=sf.Rot2.compose, config=config)

            self.assertNotEqual(
                SimilarityIndex.from_codegen(co_rot3), SimilarityIndex.from_codegen(co_rot2)
            )

        with self.subTest(msg="Sparse outputs must match to be similar"):
            inputs = Values(arg=sf.Rot3.symbolic("a"))
            outputs = Values(out=sf.Matrix(inputs["arg"].to_storage()))

            co_dense = codegen.Codegen(
                inputs=inputs, outputs=outputs, config=codegen.CppConfig(), name="sparse_test"
            )

            co_sparse = codegen.Codegen(
                inputs=inputs,
                outputs=outputs,
                config=codegen.CppConfig(),
                name="sparse_test",
                sparse_matrices=["out"],
            )

            self.assertNotEqual(
                SimilarityIndex.from_codegen(co_sparse), SimilarityIndex.from_codegen(co_dense)
            )

        with self.subTest(msg="The return value must match to be similar"):
            inputs = Values(arg=sf.Rot3.symbolic("a"))
            outputs = Values(out=inputs["arg"].inverse())

            co_arg = codegen.Codegen(
                inputs=inputs, outputs=outputs, config=codegen.CppConfig(), name="return_test"
            )
            co_out = codegen.Codegen(
                inputs=inputs,
                outputs=outputs,
                config=codegen.CppConfig(),
                name="return_test",
                return_key="out",
            )

            self.assertNotEqual(
                SimilarityIndex.from_codegen(co_arg), SimilarityIndex.from_codegen(co_out)
            )

        with self.subTest(msg="If output is different, then they are dissimilar"):
            inputs = Values(arg=sf.Rot3.symbolic("a"))

            co_1 = codegen.Codegen(
                inputs=inputs,
                outputs=Values(out=inputs["arg"]),
                config=codegen.CppConfig(),
                name="output_test",
            )
            co_2 = codegen.Codegen(
                inputs=inputs,
                outputs=Values(out=inputs["arg"].inverse()),
                config=codegen.CppConfig(),
                name="output_test",
            )

            self.assertNotEqual(
                SimilarityIndex.from_codegen(co_1), SimilarityIndex.from_codegen(co_2)
            )


if __name__ == "__main__":
    TestCase.main()
