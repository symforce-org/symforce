"""
Locate and re-export cc_sym as symforce.cc_sym
"""
import os
import sys

from symforce import path_util

sys.path.append(os.fspath(path_util.cc_sym_install_dir()))

from cc_sym import *
