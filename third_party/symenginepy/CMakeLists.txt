cmake_minimum_required(VERSION 2.8.12)
project(python_wrapper)

set(CMAKE_PREFIX_PATH ${SymEngine_DIR} ${CMAKE_PREFIX_PATH})
find_package(SymEngine 0.7.0 REQUIRED CONFIG
             PATH_SUFFIXES lib/cmake/symengine cmake/symengine CMake/)
message("SymEngine_DIR : " ${SymEngine_DIR})
message("SymEngine Version : " ${SymEngine_VERSION})
set(CMAKE_BUILD_TYPE ${SYMENGINE_BUILD_TYPE})
set(CMAKE_CXX_FLAGS_RELEASE ${SYMENGINE_CXX_FLAGS_RELEASE})
set(CMAKE_CXX_FLAGS_DEBUG ${SYMENGINE_CXX_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYMENGINE_CXX_FLAGS}")
include_directories(${SYMENGINE_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
find_package(Python REQUIRED)
find_package(Cython REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})

if (MINGW AND ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8"))
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DMS_WIN64")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DMS_WIN64")
endif()

if (MINGW AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    try_compile(CHECK_PYTHON_HYPOT "${CMAKE_CURRENT_BINARY_DIR}/" "${CMAKE_SOURCE_DIR}/cmake/check_python_hypot.cpp")
    if (NOT ${CHECK_PYTHON_HYPOT})
        # include cmath before all headers
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -include cmath")
    endif()
endif()

include(CheckTypeSize)
check_type_size("long double" SYMENGINE_SIZEOF_LONG_DOUBLE)

if (HAVE_SYMENGINE_LLVM AND  SYMENGINE_SIZEOF_LONG_DOUBLE GREATER "8" AND CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set (HAVE_SYMENGINE_LLVM_LONG_DOUBLE True)
else ()
    set (HAVE_SYMENGINE_LLVM_LONG_DOUBLE False)
endif ()

foreach (PKG MPC MPFR PIRANHA FLINT LLVM)
    if ("${HAVE_SYMENGINE_${PKG}}" STREQUAL "")
        set(HAVE_SYMENGINE_${PKG} False)
    endif()
endforeach()

message("CMAKE_SYSTEM_PROCESSOR              : ${CMAKE_SYSTEM_PROCESSOR}")
message("CMAKE_BUILD_TYPE                    : ${CMAKE_BUILD_TYPE}")
message("CMAKE_CXX_FLAGS                     : ${CMAKE_CXX_FLAGS}")
message("CMAKE_CXX_FLAGS_RELEASE             : ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_CXX_FLAGS_DEBUG               : ${CMAKE_CXX_FLAGS_DEBUG}")
message("HAVE_SYMENGINE_MPFR                 : ${HAVE_SYMENGINE_MPFR}")
message("HAVE_SYMENGINE_MPC                  : ${HAVE_SYMENGINE_MPC}")
message("HAVE_SYMENGINE_PIRANHA              : ${HAVE_SYMENGINE_PIRANHA}")
message("HAVE_SYMENGINE_FLINT                : ${HAVE_SYMENGINE_FLINT}")
message("HAVE_SYMENGINE_LLVM                 : ${HAVE_SYMENGINE_LLVM}")
message("HAVE_SYMENGINE_LLVM_LONG_DOUBLE     : ${HAVE_SYMENGINE_LLVM_LONG_DOUBLE}")
message("SYMENGINE_COPY_EXTENSION            : ${SYMENGINE_COPY_EXTENSION}")

message("Copying source of python wrappers into: ${CMAKE_CURRENT_BINARY_DIR}")
file(COPY symengine/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/symengine)

add_subdirectory(symengine)
