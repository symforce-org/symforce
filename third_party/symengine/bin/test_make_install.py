from __future__ import print_function
from os.path import join, relpath
import sys
from glob import glob

# First argument is where symengine header files are installed.
install_dir = sys.argv[1]

# Second argument is Folder containing symengine header files.
symengine_dir = sys.argv[2]

installed_files = set([relpath(x, install_dir) for x in glob(join(install_dir, '*.h'))])
all_files = set([relpath(x, symengine_dir) for x in glob(join(symengine_dir, '*.h'))])

difference = all_files - installed_files

if len(difference) == 0:
    print('All files are installed!')
    exit(0)
else:
    for fl in difference:
        print('%s is not installed!' % fl)
    exit(1)
