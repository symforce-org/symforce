project(catch)

set(SRC
    catch.cpp
)

add_library(catch STATIC ${SRC})
target_link_libraries(catch symengine)
