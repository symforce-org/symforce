import logging
import os
import tempfile

from symforce import logger
from symforce import python_util
from symforce.test_util import TestCase
from symforce.codegen import CodegenMode
from symforce.codegen import sym_util_package_codegen


SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__))


class SymforceSymUtilCodegenTest(TestCase):
    """
    Generate C++ utils
    """

    def test_codegen_cpp(self):
        # type: () -> None
        """
        Generate typedefs.h
        """
        output_dir = tempfile.mkdtemp(prefix="sf_opt_codegen_test_", dir="/tmp")
        logger.debug("Creating temp directory: {}".format(output_dir))

        try:
            sym_util_package_codegen.generate(mode=CodegenMode.CPP, output_dir=output_dir)

            self.compare_or_update_directory(
                actual_dir=os.path.join(output_dir, "sym", "util"),
                expected_dir=os.path.join(SYMFORCE_DIR, "gen", "cpp", "sym", "util"),
            )

        finally:
            if logger.level != logging.DEBUG:
                python_util.remove_if_exists(output_dir)


if __name__ == "__main__":
    TestCase.main()
