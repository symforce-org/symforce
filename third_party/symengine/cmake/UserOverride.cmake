# This overrides the default CMake Debug and Release compiler options.
# The user can still specify different options by setting the
# CMAKE_CXX_FLAGS_[RELEASE,DEBUG] variables (on the command line or in the
# CMakeList.txt). This files serves as better CMake defaults and should only be
# modified if the default values are to be changed. Project specific compiler
# flags should be set in the CMakeList.txt by setting the CMAKE_CXX_FLAGS_*
# variables.

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # g++
    set(common "-Wall -Wextra -Wno-unused-parameter -fno-common")
    set(CMAKE_CXX_FLAGS_RELEASE_INIT "${common} -O3 -funroll-loops")
    set(CMAKE_CXX_FLAGS_DEBUG_INIT   "${common} -g -ggdb")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # icpc
    set(common "-Wall -fno-common")
    set(CMAKE_CXX_FLAGS_RELEASE_INIT "${common} -xHOST -O3")
    set(CMAKE_CXX_FLAGS_DEBUG_INIT   "${common} -g -O0")
elseif (CMAKE_CXX_COMPILER_ID MATCHES Clang)
    # clang
    set(common "-Wall -Wextra -Wno-unused-parameter")
    set(CMAKE_CXX_FLAGS_RELEASE_INIT "${common} -O3 -funroll-loops")
    set(CMAKE_CXX_FLAGS_DEBUG_INIT   "${common} -g -ggdb")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
    # pgcpp
endif ()
