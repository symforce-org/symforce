"""
General python utilities.
"""
import os
import re
import shutil
import subprocess

from symforce import logger


def remove_if_exists(path):
    """
    Delete a file or directory if it exists.

    Args:
        path (str):
    """
    if not os.path.exists(path):
        logger.debug("Doesn't exist: {}".format(path))
        return
    elif os.path.isdir(path):
        logger.debug("Removing directory: {}".format(path))
        shutil.rmtree(path)
    else:
        logger.debug("Removing file: {}".format(path))
        os.remove(path)


def execute_subprocess(cmd, *args, **kwargs):
    """
    Execute subprocess and log command as well as stdout/stderr.

    Args:
        cmd (str or iterable(str)):

    Returns:
        subprocess.CalledProcessError: If the return code is nonzero
    """
    cmd_str = " ".join(cmd) if isinstance(cmd, (tuple, list)) else cmd
    logger.info("Subprocess: {}".format(cmd_str))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, *args, **kwargs)
    (stdout, other) = proc.communicate()
    logger.info(stdout)

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, stdout)


def camelcase_to_snakecase(s):
    """
    Convert CamelCase -> snake_case.

    Args:
        s (str):

    Returns:
        str:
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
