project(teuchos)

set(SRC
    Teuchos_dyn_cast.cpp
    Teuchos_Ptr.cpp
    Teuchos_RCPNode.cpp
    Teuchos_TestForException.cpp
    Teuchos_TypeNameTraits.cpp
    Teuchos_stacktrace.cpp
)

set(HEADERS
    Teuchos_any.hpp
    Teuchos_Assert.hpp
    Teuchos_ConfigDefs.hpp
    Teuchos_ConstTypeTraits.hpp
    Teuchos_DLLExportMacro.h
    Teuchos_dyn_cast.hpp
    Teuchos_ENull.hpp
    Teuchos_Exceptions.hpp
    Teuchos_getBaseObjVoidPtr.hpp
    Teuchos_map.hpp
    Teuchos_NullIteratorTraits.hpp
    Teuchos_Ptr.hpp
    Teuchos_PtrDecl.hpp
    Teuchos_RCP.hpp
    Teuchos_RCPDecl.hpp
    Teuchos_RCPNode.hpp
    Teuchos_stacktrace.hpp
    Teuchos_TestForException.hpp
    Teuchos_toString.hpp
    Teuchos_TypeNameTraits.hpp
)

if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  # Must suppress strick aliasing warnings for this file!
  SET_SOURCE_FILES_PROPERTIES(
    Teuchos_stacktrace.cpp
    PROPERTIES COMPILE_FLAGS -fno-strict-aliasing
    )
endif()

# Configure Teuchos using our CMake options:
configure_file(Teuchos_config.h.in Teuchos_config.h)
# Include the config file:
include_directories(BEFORE ${teuchos_BINARY_DIR})

add_library(teuchos STATIC ${SRC})


include(GNUInstallDirs)

install(TARGETS teuchos
            EXPORT SymEngineTargets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

if (NOT WITH_SYMENGINE_RCP)
    install(FILES ${HEADERS} "${teuchos_BINARY_DIR}/Teuchos_config.h"
        DESTINATION include/symengine/utilities/teuchos)
endif()
