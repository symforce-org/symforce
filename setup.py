# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import typing as T

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

SOURCE_DIR = Path(__file__).resolve().parent


class CMakeExtension(Extension):
    """
    CMake extension type.
    """

    def __init__(self, name: str):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    """
    Custom extension builder that runs CMake.
    """

    def run(self) -> None:
        # Test for CMake
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as ex:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            ) from ex

        build_temp_path = Path(self.build_temp)
        build_directory = build_temp_path.resolve()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_directory}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        # NOTE(hayk): Don't build tests or examples through pip.
        cmake_args += [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DSYMFORCE_BUILD_TESTS=OFF",
            "-DSYMFORCE_BUILD_EXAMPLES=OFF",
        ]

        # Assuming Makefiles
        build_args += ["--", f"-j{multiprocessing.cpu_count()}"]

        self.build_args = build_args

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )

        build_temp_path.mkdir(parents=True, exist_ok=True)

        # CMakeLists.txt is in the same directory as this setup.py file
        print("-" * 10, "Running CMake prepare", "-" * 40)
        subprocess.run(
            ["cmake", str(SOURCE_DIR)] + cmake_args, cwd=self.build_temp, env=env, check=True
        )

        print("-" * 10, "Building extensions", "-" * 40)
        cmake_cmd = ["cmake", "--build", "."] + self.build_args
        subprocess.run(cmake_cmd, cwd=self.build_temp, check=True)

        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext: CMakeExtension) -> None:
        build_temp = Path(self.build_temp).resolve()
        dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()  # type: ignore[attr-defined]
        source_path = build_temp / "pybind" / self.get_ext_filename(ext.name)  # type: ignore[attr-defined]
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(source_path, dest_path)


class InstallWithExtras(install):
    """
    Custom install step that:
        1) Installs symenginepy so it can be imported
        2) Installs additional shared libraries needed by cc_sym (e.g. libsymforce_cholesky,
           libmetis)
        3) Installs lcmtypes python package
    """

    def run(self) -> None:
        super().run()

        build_dir = Path(self.distribution.get_command_obj("build_ext").build_temp)  # type: ignore[attr-defined]

        # Install symengine
        # TODO(aaron): This is pretty jank, but it works in the nominal case
        subprocess.run(
            [
                sys.executable,
                str(SOURCE_DIR / "third_party" / "symenginepy" / "setup.py"),
                "install",
                f"--prefix={self.prefix}",
            ],
            # TODO(aaron): This seems pretty brittle, but the only way I could get symenginepy to
            # install was to run this from the same directory we do during the build
            cwd=build_dir / "symenginepy-prefix" / "src" / "symenginepy-build",
            check=True,
        )

        # Configure with install prefix
        subprocess.run(
            ["cmake", str(SOURCE_DIR), f"-DCMAKE_INSTALL_PREFIX={self.prefix}"],
            cwd=build_dir,
            check=True,
        )

        # Install other libraries needed for cc_sym.so
        subprocess.run(
            [
                "cmake",
                "--build",
                ".",
                "--target",
                "install",
            ],
            cwd=build_dir,
            check=True,
        )

        # Install lcmtypes - this might be super jank
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "lcmtypes/python2.7",
                f"--prefix={self.prefix}",
            ],
            # TODO(aaron): This seems pretty brittle, but the only way I could get symenginepy to
            # install was to run this from the same directory we do during the build
            cwd=build_dir,
            check=True,
        )


def symforce_version() -> str:
    # We can't import the version here, so we have to do some text parsing
    import re

    version_file_contents = (Path(__file__).parent / "symforce" / "_version.py").read_text()
    version_match = re.search(r'^version = "(.+)"$', version_file_contents, flags=re.MULTILINE)
    assert version_match is not None
    return version_match.group(1)


setup_requirements = [
    "setuptools",
    "wheel",
    "pip",
    "cmake>=3.17",
    "cython>=0.19.1",
    f"skymarshal @ file://localhost/{SOURCE_DIR}/third_party/skymarshal",
]

docs_requirements = [
    "ipykernel",
    # nbconvert depends on this, but doesn't specify the dependency
    "ipython-genutils",
    "matplotlib",
    "myst-parser",
    "nbsphinx",
    "nbstripout",
    "Sphinx",
    # sphinx-autodoc-typehints >=1.15 contains a bug causing it to crash parsing our typing.py
    "sphinx-autodoc-typehints<1.15",
    "sphinx-rtd-theme",
    "breathe",
]

cmdclass: T.Dict[str, T.Any] = dict(build_ext=CMakeBuild, install=InstallWithExtras)

setup(
    name="symforce",
    version=symforce_version(),
    author="Skydio, Inc",
    author_email="hayk@skydio.com",
    description="Fast symbolic computation, code generation, and nonlinear optimization for robotics",
    keywords="python computer-vision cpp robotics optimization structure-from-motion motion-planning code-generation slam autonomous-vehicles symbolic-computation",
    license="Apache 2.0",
    license_file="LICENSE",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/symforce-org/symforce",
    project_urls={
        "Bug Tracker": "https://github.com/symforce-org/symforce/issues",
        "Source": "https://github.com/symforce-org/symforce",
    },
    # For a list of valid classifiers see https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Embedded Systems",
        "Topic :: Education :: Computer Aided Instruction (CAI)",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    # -------------------------------------------------------------------------
    # Build info
    # -------------------------------------------------------------------------
    # Minimum Python version
    python_requires=">=3.8",
    # Find all packages in the directory
    packages=find_packages(),
    # Override the extension builder with our cmake class
    cmdclass=cmdclass,
    # Build C++ extension module
    ext_modules=[CMakeExtension("cc_sym")],
    # Barebones packages needed to run symforce
    install_requires=[
        "black",
        "clang-format",
        "graphviz",
        "jinja2",
        "numpy",
        f"skymarshal @ file://localhost/{SOURCE_DIR}/third_party/skymarshal",
        "sympy~=1.10.0",
        f"sym @ file://localhost/{SOURCE_DIR}/gen/python",
    ],
    setup_requires=setup_requirements,
    extras_require={
        "docs": docs_requirements,
        "dev": docs_requirements
        + [
            "click~=8.0.4",  # Required by black, but not specified by our version of black
            "argh",
            "black[jupyter]==21.12b0",
            "coverage",
            "jinja2~=3.0.3",
            "mypy==0.910",
            "pip-tools",
            "pybind11-stubgen",
            "pylint",
            "scipy",
            "types-jinja2",
            "types-pygments",
            "types-requests",
            "types-setuptools",
        ],
        "_setup": setup_requirements,
    },
    # Not okay to zip
    zip_safe=False,
)
