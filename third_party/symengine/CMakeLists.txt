cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_USER_MAKE_RULES_OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/UserOverride.cmake)

project(symengine)

set(SYMENGINE_MAJOR_VERSION 0)
set(SYMENGINE_MINOR_VERSION 7)
set(SYMENGINE_PATCH_VERSION 0)
set(SYMENGINE_VERSION ${SYMENGINE_MAJOR_VERSION}.${SYMENGINE_MINOR_VERSION}.${SYMENGINE_PATCH_VERSION})

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include(CheckCXXCompilerFlag)

# Make sure that CMAKE_BUILD_TYPE is either Debug or Release:
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release
        CACHE STRING "Build type (Debug, Release)" FORCE)
endif ()
if (NOT (CMAKE_BUILD_TYPE STREQUAL "Debug" OR
        CMAKE_BUILD_TYPE STREQUAL "Release"))
    message("${CMAKE_BUILD_TYPE}")
    message(FATAL_ERROR "CMAKE_BUILD_TYPE must be one of: Debug, Release (current value: '${CMAKE_BUILD_TYPE}')")
endif ()

set(BUILD_FOR_DISTRIBUTION no
    CACHE BOOL "Building SymEngine for a distribution")

#Boost.multiprecision build segfaults if -O3 optimization is on
#which it is by default for cmake Release mode. Remove -O3 from flags.
if (("${INTEGER_CLASS}" MATCHES "boostmp") AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    string(REPLACE "-O3" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# Enable C++11 support in all compilers. SymEngine will not compile unless
# the C++11 support is enabled.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel|SunPRO")
    set(CXX11_OPTIONS "-std=c++11")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
    # pgcpp
    set(CXX11_OPTIONS "--gnu --c++11 -D__GXX_EXPERIMENTAL_CXX0X__")
endif ()
set(CMAKE_CXX_FLAGS "${CXX11_OPTIONS} ${CMAKE_CXX_FLAGS}")

# Add proper -fPIC and MINGW handling
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (NOT MINGW)
        set(common "-fPIC")
    else()
        set(common "-D_hypot=hypot -Doff64_t=_off64_t")
        if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
            set(common "${common} -DMS_WIN64")
        endif()
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${common}")
endif ()
if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set(common "-fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${common}")
endif ()

# Check c++11 support
try_compile(CXX11 "${CMAKE_CURRENT_BINARY_DIR}/cxx" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/checkcxx11.cpp"
	CMAKE_FLAGS "-DCOMPILE_DEFINITIONS=${CMAKE_CXX_FLAGS}" OUTPUT_VARIABLE CXX11_ERROR_LOG)
if (NOT ${CXX11})
	message(FATAL_ERROR "Compiler does not support C++11 constructs. \n"
		"CXX11 Error Log : \n${CXX11_ERROR_LOG}")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    CHECK_CXX_COMPILER_FLAG("-Wno-implicit-fallthrough" NO_WARN_IMPLICIT_FALL_THROUGH_FLAG)
endif()

# check if linker supports exclude-libs
if (NOT(MSVC))
    try_compile(EXCLUDE_LIBS "${CMAKE_CURRENT_BINARY_DIR}/cxx" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/checkcxx11.cpp"
        CMAKE_FLAGS "-DCOMPILE_DEFINITIONS=${CMAKE_CXX_FLAGS}"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${GMP_INCLUDE_DIRS}"
        CMAKE_FLAGS "-DLINK_LIBRARIES=-Wl,--exclude-libs,ALL"
        OUTPUT_VARIABLE EXCLUDE_LIBS_ERROR_LOG)
endif()
if (${EXCLUDE_LIBS})
    message(STATUS "Linker supports --exclude-libs")
else()
    message(STATUS "Linker does not support --exclude-libs")
endif()

if (NOT(MSVC) AND NOT "${EXCLUDE_LIBS}" STREQUAL "TRUE")
    try_compile(UNEXPORTED_SYMBOL "${CMAKE_CURRENT_BINARY_DIR}/cxx" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/checkcxx11.cpp"
        CMAKE_FLAGS "-DCOMPILE_DEFINITIONS=${CMAKE_CXX_FLAGS}"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${GMP_INCLUDE_DIRS}"
        CMAKE_FLAGS "-DLINK_LIBRARIES=-Wl,-unexported_symbol,dummy"
        OUTPUT_VARIABLE UNEXPORTED_SYMBOL_ERROR_LOG)
    if (${UNEXPORTED_SYMBOL})
        message(STATUS "Linker supports -unexported_symbol")
    else()
        message(STATUS "Linker does not support -unexported_symbol")
    endif()
endif()

include(CheckTypeSize)
check_type_size("long double" SYMENGINE_SIZEOF_LONG_DOUBLE)

set(HAVE_GCC_ABI_DEMANGLE yes
    CACHE BOOL "Build with C++ name demangling support")

set(HAVE_C_FUNCTION_NOT_FUNC no
    CACHE BOOL "C compiler supports __FUNCTION__ but not __func__")

set(HAVE_DEFAULT_CONSTRUCTORS yes
    CACHE BOOL "C++ compiler supports default constructors")

set(HAVE_SYMENGINE_NOEXCEPT yes
    CACHE BOOL "C++ compiler supports noexcept specifier")

set(HAVE_SYMENGINE_IS_CONSTRUCTIBLE yes
    CACHE BOOL "C++ compiler supports std::is_constructible")

set(HAVE_SYMENGINE_RESERVE yes
    CACHE BOOL "C++ compiler supports std::unordered_map<>::reserve()")

if (MSVC)
    # MSVC doesn't have cxxabi.h
    set(HAVE_GCC_ABI_DEMANGLE no)

    # MSVC only supports __func__ in a C++ mode, not a C mode
    set(HAVE_C_FUNCTION_NOT_FUNC yes)

    # MSVC does not support =default for constructors
    set(HAVE_DEFAULT_CONSTRUCTORS no)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
    set(HAVE_DEFAULT_CONSTRUCTORS no)
    set(HAVE_SYMENGINE_RESERVE no)
endif()

try_compile(HAVE_SYMENGINE_STD_TO_STRING "${CMAKE_CURRENT_BINARY_DIR}/cxx" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/checkstdtostring.cpp"
	CMAKE_FLAGS "-DCOMPILE_DEFINITIONS=${CMAKE_CXX_FLAGS}")

if ((CMAKE_CXX_COMPILER_ID MATCHES Clang) AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    try_compile(CHECK_CLANG "${CMAKE_CURRENT_BINARY_DIR}/cxx" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/checkclang.cpp")
    if (NOT ${CHECK_CLANG})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__extern_always_inline=inline" )
    endif()
endif()

# Cotire
if ((CMAKE_VERSION VERSION_GREATER 2.8.11) AND
    (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|MSVC"))
    set(WITH_COTIRE yes CACHE BOOL "Use cotire to speed up build")
else()
    set(WITH_COTIRE no CACHE BOOL "Use cotire to speed up build")
endif()

set(INTEGER_CLASS "gmp"
    CACHE STRING "Integer class for symengine. Either gmp, gmpxx, flint or piranha")

# Virtual TypeID
set(WITH_VIRTUAL_TYPEID no CACHE BOOL "Use virtual TypeID getter")
set(WITH_SYMENGINE_VIRTUAL_TYPEID False)

# Piranha
set(WITH_PIRANHA no CACHE BOOL "Build with Piranha")

# FLINT
set(WITH_FLINT no CACHE BOOL "Build with Flint")
set(HAVE_SYMENGINE_FLINT False)

# Boost
set(WITH_BOOST no CACHE BOOL "Build with Boost")

set(HAVE_SYMENGINE_GMP yes)
if ("${INTEGER_CLASS}" MATCHES "boostmp")
    set(SYMENGINE_INTEGER_CLASS "BOOSTMP")
    set(WITH_BOOST yes)
    set(HAVE_SYMENGINE_GMP no)
elseif ("${INTEGER_CLASS}" MATCHES "piranha")
    set(SYMENGINE_INTEGER_CLASS "PIRANHA")
    set(WITH_PIRANHA yes)
elseif ("${INTEGER_CLASS}" MATCHES "flint")
    set(SYMENGINE_INTEGER_CLASS "FLINT")
    set(WITH_FLINT yes)
elseif ("${INTEGER_CLASS}" MATCHES "gmpxx")
    if (WITH_PIRANHA)
        message(WARNING "SymEngine can't be built with both INTEGER_CLASS=gmpxx and WITH_PIRANHA=yes")
        set(SYMENGINE_INTEGER_CLASS "GMP")
    else()
        set(SYMENGINE_INTEGER_CLASS "GMPXX")
        set(WITH_GMPXX yes)
    endif()
else()
    set(SYMENGINE_INTEGER_CLASS "GMP")
endif()

# GMP
if (HAVE_SYMENGINE_GMP)
    find_package(GMP REQUIRED)
    include_directories(SYSTEM ${GMP_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${GMP_TARGETS})
    set(PKGS ${PKGS} "GMP")

    # Check gmpxx
    if (NOT(MSVC) AND WITH_GMPXX)
        try_compile(GMPXX "${CMAKE_CURRENT_BINARY_DIR}/cxx" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/checkgmpxx.cpp"
            CMAKE_FLAGS "-DCOMPILE_DEFINITIONS=${CMAKE_CXX_FLAGS}"
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${GMP_INCLUDE_DIRS}"
            CMAKE_FLAGS "-DLINK_LIBRARIES=${GMP_LIBRARIES}"
            OUTPUT_VARIABLE GMPXX_ERROR_LOG)
        if (NOT ${GMPXX})
            message(FATAL_ERROR "GMP library being linked is not supported by CMAKE_CXX_COMPILER used. \n"
                "Recompile GMP with C++ support using ${CMAKE_CXX_COMPILER}, remove CMakeCache.txt and try again.\n"
                "Error Log : \n${GMPXX_ERROR_LOG}")
        endif()
    endif()
endif()


set(MSVC_USE_MT yes CACHE BOOL "Use MT flags when compiling in MSVC")
set(MSVC_WARNING_LEVEL 1 CACHE STRING "MSVC warning level")

if (MSVC)
    if (MSVC_USE_MT)
	    foreach(CompilerFlag CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE)
	      string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
	    endforeach()
    endif()
    foreach(CompilerFlag CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE)
      set(${CompilerFlag} "${${CompilerFlag}} /W${MSVC_WARNING_LEVEL}")
    endforeach()
endif()

if (BUILD_SHARED_LIBS)
    if (WIN32)
        if (MSVC)
            # Use CMake 3.4 feature
            set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS yes)
        else()
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--export-all-symbols")
        endif()
    endif()
endif()

# ECM
set(WITH_ECM no
    CACHE BOOL "Build with ECM (libecm-dev) support")

if (WITH_ECM)
    find_package(ECM REQUIRED)
    include_directories(SYSTEM ${ECM_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${ECM_TARGETS})
    set(HAVE_SYMENGINE_ECM yes)
    set(PKGS ${PKGS} "ECM")
endif()

# PRIMESIEVE
set(WITH_PRIMESIEVE no
    CACHE BOOL "Build with Primesieve")

if (WITH_PRIMESIEVE)
    find_package(PRIMESIEVE REQUIRED)
    include_directories(SYSTEM ${PRIMESIEVE_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${PRIMESIEVE_TARGETS})
    set(HAVE_SYMENGINE_PRIMESIEVE yes)
    set(PKGS ${PKGS} "PRIMESIEVE")
endif()

# ARB
set(WITH_ARB no
    CACHE BOOL "Build with Arb")
set(HAVE_SYMENGINE_ARB False)

# MPFR
set(WITH_MPFR no
    CACHE BOOL "Build with MPFR")
set(HAVE_SYMENGINE_MPFR False)

# MPC
set(WITH_MPC no
    CACHE BOOL "Build with MPC")
set(HAVE_SYMENGINE_MPC False)

# Virtual TypeID
if (WITH_VIRTUAL_TYPEID)
    set(WITH_SYMENGINE_VIRTUAL_TYPEID True)
endif()

# Parser
set(WITH_GENERATE_PARSER no CACHE BOOL "Generate parser files")
if (WITH_GENERATE_PARSER)
    find_program(BISON_EXECUTABLE bison)
    find_program(RE2C_EXECUTABLE re2c)

    if ("${BISON_EXECUTABLE}" STREQUAL "BISON_EXECUTABLE-NOTFOUND")
        message(FATAL_ERROR "bison not found!")
    endif()
    if ("${RE2C_EXECUTABLE}" STREQUAL "RE2C_EXECUTABLE-NOTFOUND")
        message(FATAL_ERROR "re2c not found!")
    endif()
endif()

if (WITH_FLINT)
    find_package(FLINT REQUIRED)
    include_directories(SYSTEM ${FLINT_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${FLINT_TARGETS})
    set(HAVE_SYMENGINE_FLINT True)
    set(PKGS ${PKGS} "FLINT")

    # Workaround for https://github.com/fredrik-johansson/arb/issues/24
    include_directories(SYSTEM ${FLINT_INCLUDE_DIRS}/flint)

    set(WITH_MPFR yes)
endif()

if (WITH_ARB)
    find_package(FLINT REQUIRED)
    include_directories(SYSTEM ${FLINT_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${FLINT_TARGETS})
    set(PKGS ${PKGS} "FLINT")

    # Workaround for https://github.com/fredrik-johansson/arb/issues/24
    include_directories(SYSTEM ${FLINT_INCLUDE_DIRS}/flint)

    set(WITH_FLINT yes)
    set(HAVE_SYMENGINE_FLINT True)
    set(WITH_MPFR yes)

    find_package(ARB REQUIRED)
    include_directories(SYSTEM ${ARB_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${ARB_TARGETS})
    set(HAVE_SYMENGINE_ARB True)
    set(PKGS ${PKGS} "ARB")
endif()

if (WITH_MPC)
    find_package(MPC REQUIRED)
    include_directories(SYSTEM ${MPC_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${MPC_TARGETS})
    set(HAVE_SYMENGINE_MPC True)
    set(WITH_MPFR yes)
    set(PKGS ${PKGS} "MPC")
endif()

# PTHREAD
set(WITH_PTHREAD no
CACHE BOOL "Build with PTHREAD")

if (WITH_PIRANHA)
    set(WITH_PTHREAD yes)
    set(WITH_MPFR yes)
    set(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

    find_package(PIRANHA REQUIRED)
    include_directories(SYSTEM ${PIRANHA_INCLUDE_DIRS})
    set(HAVE_SYMENGINE_PIRANHA yes)
    set(PKGS ${PKGS} "PIRANHA")
    set(BOOST_COMPONENTS ${BOOST_COMPONENTS} serialization iostreams)
    set(WITH_BOOST yes)
endif()

if (WITH_PTHREAD)
    find_package(PTHREAD REQUIRED)
    include_directories(SYSTEM ${PTHREAD_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${PTHREAD_TARGETS})
    set(HAVE_SYMENGINE_PTHREAD yes)
    set(PKGS ${PKGS} "PTHREAD")
endif()

if (WITH_MPFR)
    find_package(MPFR REQUIRED)
    include_directories(SYSTEM ${MPFR_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${MPFR_TARGETS})
    set(HAVE_SYMENGINE_MPFR True)
    set(PKGS ${PKGS} "MPFR")
endif()

# PTHREAD
set(WITH_PTHREAD no
    CACHE BOOL "Build with PTHREAD")

if (WITH_PIRANHA)
    set(WITH_PTHREAD yes)
    set(WITH_MPFR yes)
    set(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

    find_package(PIRANHA REQUIRED)
    include_directories(SYSTEM ${PIRANHA_INCLUDE_DIRS})
    set(HAVE_SYMENGINE_PIRANHA yes)
    set(PKGS ${PKGS} "PIRANHA")
    set(BOOST_COMPONENTS ${BOOST_COMPONENTS} serialization iostreams)
    set(WITH_BOOST yes)
endif()

# LLVM
set(WITH_LLVM no
    CACHE BOOL "Build with LLVM")

if (WITH_LLVM)
    set(SYMENGINE_LLVM_COMPONENTS asmparser core executionengine instcombine mcjit native nativecodegen scalaropts vectorize support transformutils)
    find_package(LLVM REQUIRED ${SYMENGINE_LLVM_COMPONENTS})
    set(LLVM_MINIMUM_REQUIRED_VERSION "3.8")
    if (LLVM_PACKAGE_VERSION LESS ${LLVM_MINIMUM_REQUIRED_VERSION})
	    message(FATAL_ERROR "LLVM version found ${LLVM_PACKAGE_VERSION} is too old.
                             Require at least ${LLVM_MINIMUM_REQUIRED_VERSION}")
    endif()
    foreach(LLVM_FLAG ${LLVM_DEFINITIONS})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_FLAG}")
    endforeach()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

    llvm_map_components_to_libnames(llvm_libs_direct ${SYMENGINE_LLVM_COMPONENTS})
    llvm_expand_dependencies(llvm_libs ${llvm_libs_direct})

    if (BUILD_SHARED_LIBS)
        set(SYMENGINE_LLVM_LINK_DOWNSTREAM_DEFAULT False)
        foreach(LLVM_LIB ${llvm_libs})
            get_target_property(${LLVM_LIB}_IMPORT_LOCATION ${LLVM_LIB} LOCATION)
            if (NOT "${${LLVM_LIB}_IMPORT_LOCATION}" MATCHES "NOTFOUND")
                if (NOT "${${LLVM_LIB}_IMPORT_LOCATION}" MATCHES ".a$|.lib$")
                    set(SYMENGINE_LLVM_LINK_DOWNSTREAM_DEFAULT True)
                endif()
            endif()
        endforeach()
    else()
        set(SYMENGINE_LLVM_LINK_DOWNSTREAM_DEFAULT True)
    endif()
    set(SYMENGINE_LLVM_LINK_DOWNSTREAM ${SYMENGINE_LLVM_LINK_DOWNSTREAM_DEFAULT}
        CACHE INTERNAL "Link to llvm in SymEngineConfig.cmake")
    if (NOT SYMENGINE_LLVM_LINK_DOWNSTREAM)
        unset(SYMENGINE_LLVM_COMPONENTS)
    endif ()
    set(LIBS ${LIBS} ${llvm_libs})
    include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
    set(HAVE_SYMENGINE_LLVM yes)
    set(PKGS ${PKGS} "LLVM")
    message("LLVM VERSION: ${LLVM_VERSION}")
endif()

# BENCHMARKS
set(BUILD_BENCHMARKS yes
    CACHE BOOL "Build SymEngine benchmarks")

set(BUILD_BENCHMARKS_NONIUS no
    CACHE BOOL "Build SymEngine Nonius benchmarks")

set(BUILD_BENCHMARKS_GOOGLE no
    CACHE BOOL "Build SymEngine Google Benchmark benchmarks")

if (BUILD_BENCHMARKS)
    if (BUILD_BENCHMARKS_NONIUS)
        if(NOT WIN32)
            set(WITH_PTHREAD yes)
        endif()
        set(WITH_BOOST yes)
    endif()
endif()

if (WITH_BOOST)
    find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${Boost_LIBRARIES})
    set(HAVE_SYMENGINE_BOOST yes)
endif()

# BFD
set(WITH_BFD no
    CACHE BOOL "Build with BFD (binutils-dev) support")

if (WITH_BFD)
    find_package(BFD REQUIRED)
    find_package(LINKH REQUIRED)
    include_directories(SYSTEM ${BFD_INCLUDE_DIRS} ${LINKH_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${BFD_TARGETS})
    set(HAVE_TEUCHOS_BFD yes)
    set(HAVE_TEUCHOS_LINK yes)

    set(HAVE_TEUCHOS_EXECINFO yes)
    find_package(EXECINFO REQUIRED)
    include_directories(SYSTEM ${EXECINFO_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${EXECINFO_TARGETS})
    set(PKGS ${PKGS} "BFD" "LINKH" "EXECINFO")
endif()

# TCMalloc
set(WITH_TCMALLOC no
    CACHE BOOL "Build with TCMalloc linked")

if (WITH_TCMALLOC)
    find_package(TCMALLOC REQUIRED)
    set(LIBS ${LIBS} ${TCMALLOC_TARGETS})

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
    endif()
    set(PKGS ${PKGS} "TCMALLOC")

endif()

# Doxygen
set(BUILD_DOXYGEN no
    CACHE BOOL "Create C++ API Doxgyen documentation.")

if (BUILD_DOXYGEN)
    find_package(Doxygen REQUIRED)
    if(DOXYGEN_FOUND)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile
            ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile @ONLY)
        add_custom_target(doc ALL
            ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc
            COMMENT "Generating API documentation with Doxygen" VERBATIM)
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/html/ DESTINATION doc)
    endif(DOXYGEN_FOUND)
endif()

# SYMENGINE_ASSERT
set(WITH_SYMENGINE_ASSERT no
    CACHE BOOL "Enable SYMENGINE_ASSERT macro")

# SYMENGINE_THREAD_SAFE
set(WITH_SYMENGINE_THREAD_SAFE "${BUILD_FOR_DISTRIBUTION}"
    CACHE BOOL "Enable SYMENGINE_THREAD_SAFE support")

# TESTS
set(BUILD_TESTS yes
    CACHE BOOL "Build SymEngine tests")

# Teuchos
set(WITH_SYMENGINE_TEUCHOS no
    CACHE BOOL "Build with teuchos")

# OpenMp
set(WITH_OPENMP no
    CACHE BOOL "Build with OpenMP")

if (WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(WITH_SYMENGINE_THREAD_SAFE yes)
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES Clang|GNU)
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -Wno-unknown-pragmas")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # In Debug mode we use Teuchos::RCP and enable debugging checks that make
    # the usage 100% safe, as long as the Teuchos guidelines are followed.
    set(WITH_SYMENGINE_RCP_DEFAULT no) # Use the Teuchos::RCP
    set(HAVE_TEUCHOS_DEBUG yes) # Enable safety checks
    set(HAVE_TEUCHOS_DEBUG_RCP_NODE_TRACING yes) # Enable safety checks

    set(WITH_SYMENGINE_ASSERT yes) # Also enable assertions
else ()
    set(WITH_SYMENGINE_RCP_DEFAULT yes)
endif()

# SYMENGINE_RCP
set(WITH_SYMENGINE_RCP ${WITH_SYMENGINE_RCP_DEFAULT}
    CACHE BOOL "Enable SYMENGINE_RCP support")

if (WITH_COVERAGE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} --coverage")
endif()

if ((NOT WITH_SYMENGINE_RCP) OR HAVE_TEUCHOS_BFD)
    set(WITH_SYMENGINE_TEUCHOS yes)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  ## References:
  ## cmake  --help-policy CMP0042
  set(CMAKE_MACOSX_RPATH ON)
endif()

if(CMAKE_INSTALL_RPATH_USE_LINK_PATH)
  ## References:
  ## https://cmake.org/Wiki/CMake_RPATH_handling#Mac_OS_X_and_the_RPATH
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  # set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  if("${isSystemDir}" STREQUAL "-1")
     set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
     set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
  endif("${isSystemDir}" STREQUAL "-1")
endif()

enable_testing()
add_subdirectory(symengine)

set(SymEngine_DIR ${PROJECT_BINARY_DIR} CACHE BOOL "SymEngine Build Directory")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})

if(WIN32 AND NOT CYGWIN)
    set(DEF_INSTALL_CMAKE_DIR CMake)
    set(DEF_INSTALL_CMAKE_DIR_REL ..)
else()
    set(DEF_INSTALL_CMAKE_DIR lib/cmake/symengine)
    set(DEF_INSTALL_CMAKE_DIR_REL ../../..)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
    "Installation directory for CMake files")

if (TARGET teuchos)
    set(SYMENGINE_TARGETS symengine teuchos)
else()
    set(SYMENGINE_TARGETS symengine)
endif()

export(TARGETS ${SYMENGINE_TARGETS} FILE "${PROJECT_BINARY_DIR}/SymEngineTargets.cmake")
foreach(PKG ${PKGS})
    set(SYMENGINE_PKG_FINDS "${SYMENGINE_PKG_FINDS}\nset(SYMENGINE_${PKG}_LIBRARIES ${${PKG}_LIBRARIES})")
    set(SYMENGINE_PKG_FINDS "${SYMENGINE_PKG_FINDS}\nset(SYMENGINE_${PKG}_INCLUDE_DIRS ${${PKG}_INCLUDE_DIRS})")
    set(SYMENGINE_PKG_FINDS "${SYMENGINE_PKG_FINDS}\nset(HAVE_SYMENGINE_${PKG} True)")
endforeach()

# ... for the build tree
set(CONF_INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
if (TARGET teuchos)
    set(CONF_INCLUDE_DIRS ${CONF_INCLUDE_DIRS}
                           ${PROJECT_SOURCE_DIR}/symengine/utilities/teuchos ${PROJECT_BINARY_DIR}/symengine/utilities/teuchos)
endif()
set(SYMENGINE_BUILD_TREE yes)
configure_file(cmake/SymEngineConfig.cmake.in
    "${PROJECT_BINARY_DIR}/SymEngineConfig.cmake" @ONLY)

# ... for the install tree
set(SYMENGINE_BUILD_TREE no)
configure_file(cmake/SymEngineConfig.cmake.in
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SymEngineConfig.cmake" @ONLY)

#include(CMakePackageConfigHelpers)
#write_basic_package_version_file(
#    "${CMAKE_CURRENT_BINARY_DIR}/SymEngineConfigVersion.cmake"
#    VERSION ${SYMENGINE_VERSION}
#    COMPATIBILITY AnyNewerVersion
#)

configure_file(cmake/SymEngineConfigVersion.cmake.in SymEngineConfigVersion.cmake @ONLY)

install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SymEngineConfig.cmake"
              ${PROJECT_BINARY_DIR}/SymEngineConfigVersion.cmake
              #cmake/FindFLINT.cmake
              #cmake/FindARB.cmake
              #cmake/FindBFD.cmake
              #cmake/FindECM.cmake
              #cmake/FindEXECINFO.cmake
              #cmake/FindFLINT.cmake
              #cmake/FindGMP.cmake
              #cmake/FindLINKH.cmake
              #cmake/FindMPC.cmake
              #cmake/FindMPFR.cmake
              #cmake/FindPIRANHA.cmake
              #cmake/FindPRIMESIEVE.cmake
              #cmake/FindPTHREAD.cmake
              #cmake/FindTCMALLOC.cmake
              #cmake/LibFindMacros.cmake
        DESTINATION ${INSTALL_CMAKE_DIR})
install(EXPORT SymEngineTargets DESTINATION ${INSTALL_CMAKE_DIR})

file(COPY ${symengine_SOURCE_DIR}/cmake/ DESTINATION ${symengine_BINARY_DIR}/cmake)

if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# At the end we print a nice summary

message("--------------------------------------------------------------------------------")
message("")
message("+-------------------------------+")
message("| Configuration results SUMMARY |")
message("+-------------------------------+")
message("")
message("CMAKE_C_COMPILER:   ${CMAKE_C_COMPILER}")
message("CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message("CMAKE_CONFIGURATION_TYPES: ${CMAKE_CONFIGURATION_TYPES}")
message("CMAKE_C_FLAGS:           ${CMAKE_C_FLAGS}")
message("CMAKE_CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
message("CMAKE_CXX_FLAGS_DEBUG:   ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message("BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")

message("INTEGER_CLASS : ${SYMENGINE_INTEGER_CLASS}")
message("HAVE_SYMENGINE_GMP: ${HAVE_SYMENGINE_GMP}")
message("WITH_SYMENGINE_ASSERT: ${WITH_SYMENGINE_ASSERT}")
message("WITH_SYMENGINE_RCP: ${WITH_SYMENGINE_RCP}")
message("WITH_SYMENGINE_TEUCHOS: ${WITH_SYMENGINE_TEUCHOS}")
if (NOT WITH_SYMENGINE_RCP)
    message("HAVE_TEUCHOS_DEBUG: ${HAVE_TEUCHOS_DEBUG}")
    message("HAVE_TEUCHOS_DEBUG_RCP_NODE_TRACING: ${HAVE_TEUCHOS_DEBUG_RCP_NODE_TRACING}")
endif ()
message("WITH_COTIRE: ${WITH_COTIRE}")
message("WITH_GENERATE_PARSER: ${WITH_GENERATE_PARSER}")
message("HAVE_GCC_ABI_DEMANGLE: ${HAVE_GCC_ABI_DEMANGLE}")
message("HAVE_C_FUNCTION_NOT_FUNC: ${HAVE_C_FUNCTION_NOT_FUNC}")
message("HAVE_DEFAULT_CONSTRUCTORS: ${HAVE_DEFAULT_CONSTRUCTORS}")
message("HAVE_SYMENGINE_NOEXCEPT: ${HAVE_SYMENGINE_NOEXCEPT}")
message("HAVE_SYMENGINE_IS_CONSTRUCTIBLE: ${HAVE_SYMENGINE_IS_CONSTRUCTIBLE}")
message("HAVE_SYMENGINE_RESERVE: ${HAVE_SYMENGINE_RESERVE}")
message("HAVE_SYMENGINE_STD_TO_STRING: ${HAVE_SYMENGINE_STD_TO_STRING}")
message("WITH_SYMENGINE_THREAD_SAFE: ${WITH_SYMENGINE_THREAD_SAFE}")
message("BUILD_TESTS: ${BUILD_TESTS}")
message("BUILD_BENCHMARKS: ${BUILD_BENCHMARKS}")
message("BUILD_BENCHMARKS_NONIUS: ${BUILD_BENCHMARKS_NONIUS}")
message("BUILD_BENCHMARKS_GOOGLE: ${BUILD_BENCHMARKS_GOOGLE}")

message("WITH_GMP: ${HAVE_SYMENGINE_GMP}")
if (HAVE_SYMENGINE_GMP)
    message("GMP_INCLUDE_DIRS: ${GMP_INCLUDE_DIRS}")
    message("GMP_LIBRARIES: ${GMP_LIBRARIES}")
endif()

message("WITH_BFD: ${WITH_BFD}")
if (WITH_BFD)
    message("BFD_INCLUDE_DIRS: ${BFD_INCLUDE_DIRS}")
    message("BFD_LIBRARIES: ${BFD_LIBRARIES}")
    message("LINKH_INCLUDE_DIRS: ${LINKH_INCLUDE_DIRS}")
    message("EXECINFO_INCLUDE_DIRS: ${EXECINFO_INCLUDE_DIRS}")
endif()

message("WITH_ECM: ${WITH_ECM}")
if (WITH_ECM)
    message("ECM_INCLUDE_DIRS: ${ECM_INCLUDE_DIRS}")
    message("ECM_LIBRARIES: ${ECM_LIBRARIES}")
endif()

message("WITH_PRIMESIEVE: ${WITH_PRIMESIEVE}")
if (WITH_PRIMESIEVE)
    message("PRIMESIEVE_INCLUDE_DIRS: ${PRIMESIEVE_INCLUDE_DIRS}")
    message("PRIMESIEVE_LIBRARIES: ${PRIMESIEVE_LIBRARIES}")
endif()

message("WITH_FLINT: ${WITH_FLINT}")
if (WITH_FLINT)
    message("FLINT_INCLUDE_DIRS: ${FLINT_INCLUDE_DIRS}")
    message("FLINT_LIBRARIES: ${FLINT_LIBRARIES}")
endif()

message("WITH_ARB: ${WITH_ARB}")
if (WITH_ARB)
    message("ARB_INCLUDE_DIRS: ${ARB_INCLUDE_DIRS}")
    message("ARB_LIBRARIES: ${ARB_LIBRARIES}")
    message("FLINT_INCLUDE_DIRS: ${FLINT_INCLUDE_DIRS}")
    message("FLINT_LIBRARIES: ${FLINT_LIBRARIES}")
endif()

message("WITH_MPFR: ${WITH_MPFR}")
if (WITH_MPFR)
    message("MPFR_INCLUDE_DIRS: ${MPFR_INCLUDE_DIRS}")
    message("MPFR_LIBRARIES: ${MPFR_LIBRARIES}")
endif()

message("WITH_PIRANHA: ${WITH_PIRANHA}")
if (WITH_PIRANHA)
    message("PIRANHA_INCLUDE_DIRS: ${PIRANHA_INCLUDE_DIRS}")
endif()

message("WITH_LLVM: ${WITH_LLVM}")
if (WITH_LLVM)
    message("LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
endif()

message("WITH_BOOST: ${WITH_BOOST}")
if (WITH_BOOST)
    message("BOOST_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message("BOOST_LIBRARIES: ${Boost_LIBRARIES}")
endif()

message("WITH_PTHREAD: ${WITH_PTHREAD}")
if (WITH_PTHREAD)
    message("PTHREAD_INCLUDE_DIRS: ${PTHREAD_INCLUDE_DIRS}")
    message("PTHREAD_LIBRARIES: ${PTHREAD_LIBRARIES}")
endif()

message("WITH_MPC: ${WITH_MPC}")
if (WITH_MPC)
    message("MPC_INCLUDE_DIRS: ${MPC_INCLUDE_DIRS}")
    message("MPC_LIBRARIES: ${MPC_LIBRARIES}")
endif()

message("WITH_TCMALLOC: ${WITH_TCMALLOC}")
if (WITH_TCMALLOC)
    message("TCMALLOC_LIBRARIES: ${TCMALLOC_LIBRARIES}")
endif()

message("WITH_OPENMP: ${WITH_OPENMP}")
message("WITH_VIRTUAL_TYPEID: ${WITH_VIRTUAL_TYPEID}")

message("")
message("--------------------------------------------------------------------------------")
message("")
