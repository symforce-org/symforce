from __future__ import print_function
from os.path import join, relpath, sep
import sys
from glob import glob

# First argument is where symengine header files are installed.
install_dir = sys.argv[1]

# Second argument is Folder containing symengine header files.
symengine_dir = sys.argv[2]


def ignore(header_file):
    ignore_folders = ["tests", "utilities"]
    return any([f + sep in header_file for f in ignore_folders])


def get_headers(folder):
    header_files = [
        relpath(header_file, folder)
        for header_file in glob(join(folder, "**/*.h"), recursive=True)
    ]
    return set([f for f in header_files if not ignore(f)])

installed_files = get_headers(install_dir)
all_files = get_headers(symengine_dir)

difference = all_files - installed_files

if len(difference) == 0:
    print("All files are installed!")
    exit(0)
else:
    for fl in difference:
        print("%s is not installed!" % fl)
    exit(1)
