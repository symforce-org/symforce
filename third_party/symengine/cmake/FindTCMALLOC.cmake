include(LibFindMacros)

libfind_library(tcmalloc tcmalloc)
set(TCMALLOC_TARGETS tcmalloc)
if (NOT TCMALLOC_LIBRARY)
    libfind_library(tcmalloc_minimal tcmalloc)
    set(TCMALLOC_LIBRARY ${TCMALLOC_MINIMAL_LIBRARY})
    set(TCMALLOC_TARGETS tcmalloc_minimal)
endif()
set(TCMALLOC_LIBRARIES ${TCMALLOC_LIBRARY})
find_package_handle_standard_args(TCMALLOC DEFAULT_MSG
    TCMALLOC_LIBRARIES)

mark_as_advanced(TCMALLOC_LIBRARY TCMALLOC_MINIMAL_LIBRARY)
