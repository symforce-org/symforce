include(LibFindMacros)

libfind_include(bfd.h bfd)
libfind_library(bfd bfd)
libfind_library(iberty iberty)
libfind_library(z z)
libfind_library(dl dl)

set(BFD_INCLUDE_DIRS ${BFD_INCLUDE_DIR})
set(BFD_LIBRARIES ${BFD_LIBRARY})
set(BFD_TARGETS bfd)
if (IBERTY_LIBRARY)
    set(BFD_LIBRARIES ${BFD_LIBRARIES} ${IBERTY_LIBRARY})
    set(BFD_TARGETS ${BFD_TARGETS} iberty)
endif(IBERTY_LIBRARY)
if (Z_LIBRARY)
    set(BFD_LIBRARIES ${BFD_LIBRARIES} ${Z_LIBRARY})
    set(BFD_TARGETS ${BFD_TARGETS} z)
endif(Z_LIBRARY)
if (DL_LIBRARY)
    set(BFD_LIBRARIES ${BFD_LIBRARIES} ${DL_LIBRARY})
    set(BFD_TARGETS ${BFD_TARGETS} dl)
endif(DL_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BFD DEFAULT_MSG
    BFD_LIBRARIES BFD_INCLUDE_DIRS)

mark_as_advanced(BFD_INCLUDE_DIR BFD_LIBRARY)
