#!/usr/bin/env bash

# Exit on error
set -e
# Echo each command
set -x

# Shippable currently does not clean the directory after previous builds
# (https://github.com/Shippable/support/issues/238), so
# we need to do it ourselves.
git clean -dfx

if [[ "$(uname)" == "Darwin"  ]]; then
    export TRAVIS_OS_NAME="osx"
else
    export TRAVIS_OS_NAME="linux"
fi

if [[ "${CC}" == "" ]]; then
    if [[ "${TRAVIS_OS_NAME}" == "osx" ]]; then
        export CC=clang
        export CXX=clang++
    else
        export CC=gcc
        export CXX=g++
    fi
fi

if [[ "${CXX}" == "" ]]; then
    if [[ "$CC" == gcc ]]; then
        export CXX=g++
    elif [[ "$CC" == clang ]]; then
        export CXX=clang++
    fi
fi
export GCOV_EXECUTABLE=gcov

if [[ "${TRAVIS_OS_NAME}" == "linux" ]] && [[ "${CC}" == "gcc" ]]; then
    if [[ "${WITH_PIRANHA}" == "yes" ]]; then
        export CC=gcc-4.8
        export CXX=g++-4.8
        export GCOV_EXECUTABLE=gcov-4.8
    elif [[ "${WITH_LATEST_GCC}" == "yes" ]]; then
        export CC=gcc-8
        export CXX=g++-8
        export GCOV_EXECUTABLE=gcov-8
    elif [[ "${WITH_GCC_6}" == "yes" ]]; then
        export CC=gcc-6
        export CXX=g++-6
        export GCOV_EXECUTABLE=gcov-6
    else
        export CC=gcc-4.7
        export CXX=g++-4.7
        export GCOV_EXECUTABLE=gcov-4.7
    fi
fi

if [[ "$WITH_LLVM" != "" && "${TRAVIS_OS_NAME}" == "linux" && "$GITHUB_ACTIONS" == "true" ]]; then
    wget http://ftp.gnu.org/gnu/binutils/binutils-2.32.tar.xz
    tar -xf binutils-2.32.tar.xz
    pushd binutils-2.32
    ./configure --disable-static --enable-shared --prefix=/usr
    make
    sudo make install
    popd
fi

export SOURCE_DIR=`pwd`
export our_install_dir="$HOME/our_usr"

if [[ ! -d $HOME/conda_root/pkgs ]]; then
    rm -rf $HOME/conda_root
    if [[ "${TRAVIS_OS_NAME}" == "osx" ]]; then
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
    else
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
    fi
    bash miniconda.sh -b -p $HOME/conda_root
fi
export PATH="$HOME/conda_root/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda config --add channels conda-forge --force
# Useful for debugging any issues with conda
conda info -a

# conda_pkgs="$conda_pkgs ccache"

if [[ "${INTEGER_CLASS}" == "boostmp" ]]; then
    conda_pkgs="$conda_pkgs boost=1.62";
else
    conda_pkgs="$conda_pkgs gmp=6.1.1";
fi

if [[ "${WITH_BENCHMARKS_NONIUS}" == "yes" ]]; then
    conda_pkgs="${conda_pkgs} boost=1.62"
fi

if [[ "${WITH_PIRANHA}" == "yes" ]]; then
    conda_pkgs="$conda_pkgs piranha=0.8 cmake=3.10.0"
fi

if [[ "${WITH_PRIMESIEVE}" == "yes" ]]; then
    conda_pkgs="$conda_pkgs primesieve=5.6.0"
fi

if [[ "${WITH_MPFR}" == "yes" ]]; then
    conda_pkgs="$conda_pkgs mpfr=3.1.4"
fi

if [[ "${WITH_MPC}" == "yes" ]]; then
    conda_pkgs="$conda_pkgs mpc=1.0.3"
fi

if [[ "${WITH_FLINT}" == "yes" ]] && [[ "${WITH_FLINT_DEV}" != "yes" ]]; then
    conda_pkgs="$conda_pkgs libflint=2.5.2"
fi

if [[ "${WITH_ARB}" == "yes" ]]; then
    conda_pkgs="$conda_pkgs arb=2.8.1"
fi

if [[ "${WITH_LLVM}" == "7.0" ]]; then
    export LLVM_DIR=/usr/lib/llvm-7/share/llvm/
    export CC=clang-7
    export CXX=clang++-7
elif [[ "${WITH_LLVM}" == "8.0" ]]; then
    export LLVM_DIR=/usr/lib/llvm-8/share/llvm/
elif [[ "${WITH_LLVM}" == "6.0" ]]; then
    export LLVM_DIR=/usr/lib/llvm-6.0/share/llvm/
elif [[ ! -z "${WITH_LLVM}" ]]; then
    conda_pkgs="$conda_pkgs llvmdev=${WITH_LLVM} cmake=3.10.0"
    export LLVM_DIR=$our_install_dir/share/llvm/
fi

if [[ "${WITH_ECM}" == "yes" ]]; then
    conda_pkgs="$conda_pkgs ecm=7.0.4"
fi

if [[ "${BUILD_DOXYGEN}" == "yes" ]]; then
    conda_pkgs="$conda_pkgs doxygen=1.8.13"
fi

if [[ "${BUILD_TUTORIALS}" == "yes" ]]; then
    conda_pkgs="$conda_pkgs jupytext papermill xeus-cling"
fi

if [[ "${CONDA_ENV_FILE}" == "" ]]; then
    conda create -q -p $our_install_dir ${conda_pkgs};
else
    conda env create -q -p $our_install_dir --file ${CONDA_ENV_FILE};
fi
source activate $our_install_dir;

if [[ "${WITH_FLINT_DEV}" == "yes" ]] && [[ "${WITH_ARB}" != "yes" ]]; then
    git clone https://github.com/wbhart/flint2;
    cd flint2 && git checkout 06defcbc52efe41a8c73496ffde9fc66941e3f0d && ./configure --prefix=$our_install_dir --with-gmp=$our_install_dir --with-mpfr=$our_install_dir && make -j8 install && cd ..;
fi

# Use ccache
# export CXX="ccache ${CXX}"
# export CC="ccache ${CC}"
# export CCACHE_DIR=$HOME/.ccache
# ccache -M 400M

cd $SOURCE_DIR;

# Since this script is getting sourced, remove error on exit
set +e
set +x
