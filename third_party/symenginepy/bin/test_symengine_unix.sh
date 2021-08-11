export PYTHON_SOURCE_DIR=`pwd`
export TEST_CPP="no"
export MAKEFLAGS="-j2"

git clone https://github.com/symengine/symengine symengine-cpp
cd symengine-cpp
export SOURCE_DIR=`pwd`
git checkout `cat ../symengine_version.txt`
cd ..

# Setup travis for C++ library
cd $SOURCE_DIR
source bin/test_symengine_unix.sh

# Setup travis for Python wrappers
cd $PYTHON_SOURCE_DIR
source bin/install_travis.sh

# Build Python wrappers and test
cd $PYTHON_SOURCE_DIR
bin/test_travis.sh

