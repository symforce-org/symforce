project(matchpycpp)

include_directories(BEFORE ${symengine_SOURCE_DIR})
include_directories(BEFORE ${symengine_BINARY_DIR})
include_directories(BEFORE ${teuchos_SOURCE_DIR})
include_directories(BEFORE ${teuchos_BINARY_DIR})

set(SRC
)

set(HEADER
    bipartite.h
    common.h
    generator_trick.h
    many_to_one.h
    substitution.h
    util.h
)

include_directories(BEFORE ${matchpycpp_BINARY_DIR})
