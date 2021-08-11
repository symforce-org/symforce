set(PYTHON_BIN python CACHE STRING "Python executable name")

execute_process(
	COMMAND ${PYTHON_BIN} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
	OUTPUT_VARIABLE PYTHON_SYS_PATH
	)
string(STRIP ${PYTHON_SYS_PATH} PYTHON_SYS_PATH)
FIND_PATH(PYTHON_INCLUDE_PATH Python.h
    PATHS ${PYTHON_SYS_PATH}
    NO_DEFAULT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    )
message(STATUS "Python include path: ${PYTHON_INCLUDE_PATH}")

set(PYTHON_INSTALL_HEADER_PATH ${PYTHON_INCLUDE_PATH}/symengine
    CACHE BOOL "Python install headers path")

execute_process(
	COMMAND ${PYTHON_BIN} -c "from distutils.sysconfig import get_config_var; print(get_config_var('LIBDIR'))"
	OUTPUT_VARIABLE PYTHON_LIB_PATH
	)
string(STRIP ${PYTHON_LIB_PATH} PYTHON_LIB_PATH)

execute_process(
	COMMAND ${PYTHON_BIN} -c "import sys; print(sys.prefix)"
	OUTPUT_VARIABLE PYTHON_PREFIX_PATH
	)

string(STRIP ${PYTHON_PREFIX_PATH} PYTHON_PREFIX_PATH)

execute_process(
	COMMAND ${PYTHON_BIN} -c "import sys; print('%s.%s' % sys.version_info[:2])"
    OUTPUT_VARIABLE PYTHON_VERSION
	)
string(STRIP ${PYTHON_VERSION} PYTHON_VERSION)
message(STATUS "Python version: ${PYTHON_VERSION}")

string(REPLACE "." "" PYTHON_VERSION_WITHOUT_DOTS ${PYTHON_VERSION})

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  FIND_LIBRARY(PYTHON_LIBRARY NAMES
        python${PYTHON_VERSION}
        python${PYTHON_VERSION}m
        python${PYTHON_VERSION_WITHOUT_DOTS}
      PATHS ${PYTHON_LIB_PATH} ${PYTHON_PREFIX_PATH}/lib ${PYTHON_PREFIX_PATH}/libs
      PATH_SUFFIXES ${CMAKE_LIBRARY_ARCHITECTURE}
      NO_DEFAULT_PATH
      NO_SYSTEM_ENVIRONMENT_PATH
  )
endif()

execute_process(
	COMMAND ${PYTHON_BIN} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
	OUTPUT_VARIABLE PYTHON_INSTALL_PATH_tmp
	)
string(STRIP ${PYTHON_INSTALL_PATH_tmp} PYTHON_INSTALL_PATH_tmp)
set(PYTHON_INSTALL_PATH ${PYTHON_INSTALL_PATH_tmp}
    CACHE BOOL "Python install path")
message(STATUS "Python install path: ${PYTHON_INSTALL_PATH}")

execute_process(
    COMMAND ${PYTHON_BIN} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_suffix.py
    OUTPUT_VARIABLE PYTHON_EXTENSION_SOABI_tmp
)
string(STRIP ${PYTHON_EXTENSION_SOABI_tmp} PYTHON_EXTENSION_SOABI_tmp)

set(PYTHON_EXTENSION_SOABI ${PYTHON_EXTENSION_SOABI_tmp}
    CACHE STRING "Suffix for python extensions")

INCLUDE(FindPackageHandleStandardArgs)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(Python DEFAULT_MSG PYTHON_LIBRARY PYTHON_INCLUDE_PATH PYTHON_INSTALL_PATH)
else ()
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(Python DEFAULT_MSG PYTHON_INCLUDE_PATH PYTHON_INSTALL_PATH)
endif ()


# Links a Python extension module.
#
# The exact link flags are platform dependent and this macro makes it possible
# to write platform independent cmakefiles. All you have to do is to change
# this:
#
# add_library(simple_wrapper SHARED ${SRC})  # Linux only
# set_target_properties(simple_wrapper PROPERTIES PREFIX "")
#
# to this:
#
# add_python_library(simple_wrapper ${SRC})  # Platform independent
#
# Full example:
#
# set(SRC
#     iso_c_utilities.f90
#     pde_pointers.f90
#     example1.f90
#     example2.f90
#     example_eigen.f90
#     simple.f90
#     simple_wrapper.c
# )
# add_python_library(simple_wrapper ${SRC})

macro(ADD_PYTHON_LIBRARY name)
    # When linking Python extension modules, a special care must be taken about
    # the link flags, which are platform dependent:
    IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        # on Mac, we need to use the "-bundle" gcc flag, which is what MODULE
        # does:
        add_library(${name} MODULE ${ARGN})
        # and "-flat_namespace -undefined suppress" link flags, that we need
        # to add by hand:
        set_property(TARGET ${name} APPEND_STRING PROPERTY
            LINK_FLAGS " -flat_namespace -undefined suppress -Wl,-exported_symbol,_PyInit_${name}")
    ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        # on Linux, we need to use the "-shared" gcc flag, which is what SHARED
        # does:
        set(PYTHON_EXTENSION_NAME ${name})
        add_library(${name} SHARED ${ARGN})
        configure_file(${CMAKE_SOURCE_DIR}/cmake/version_script.txt
            ${CMAKE_CURRENT_BINARY_DIR}/version_script_${name}.txt @ONLY)
        set_property(TARGET ${name} APPEND_STRING PROPERTY
            LINK_FLAGS " -Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/version_script_${name}.txt")
    ELSE()
        add_library(${name} SHARED ${ARGN})
    ENDIF()
    set_target_properties(${name} PROPERTIES PREFIX "")
    set_target_properties(${name} PROPERTIES OUTPUT_NAME "${name}${PYTHON_EXTENSION_SOABI}")
    IF(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
        target_link_libraries(${name} ${PYTHON_LIBRARY})
        set_target_properties(${name} PROPERTIES SUFFIX ".pyd")
    ENDIF()
endmacro(ADD_PYTHON_LIBRARY)
