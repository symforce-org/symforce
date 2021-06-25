import logging
import os
import tempfile

from symforce.codegen import template_util
from symforce import logger
from symforce import python_util
from symforce.test_util import TestCase

SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__))


class SymforceLcmCodegenTest(TestCase):
    def test_generate_lcm(self) -> None:
        output_dir = tempfile.mkdtemp(prefix="sf_lcm_codegen_test", dir="/tmp")
        logger.debug(f"Creating temp directory: {output_dir}")

        try:
            template_util.render_template(
                os.path.join(template_util.LCM_TEMPLATE_DIR, "symforce_types.lcm.jinja"),
                data={},
                output_path=os.path.join(output_dir, "symforce_types.lcm"),
            )

            self.compare_or_update_file(
                new_file=os.path.join(output_dir, "symforce_types.lcm"),
                path=os.path.join(SYMFORCE_DIR, "lcmtypes", "symforce_types.lcm"),
            )

        finally:
            if logger.level != logging.DEBUG:
                python_util.remove_if_exists(output_dir)


if __name__ == "__main__":
    TestCase.main()
