include(LibFindMacros)

libfind_include(mpc.h mpc)
libfind_library(mpc mpc)

set(MPC_LIBRARIES ${MPC_LIBRARY})
set(MPC_INCLUDE_DIRS ${MPC_INCLUDE_DIR})
set(MPC_TARGETS mpc)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPC DEFAULT_MSG MPC_LIBRARIES
    MPC_INCLUDE_DIRS)

mark_as_advanced(MPC_INCLUDE_DIR MPC_LIBRARY)
