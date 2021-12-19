cmake_minimum_required(VERSION 3.17)  # For CMAKE_CURRENT_FUNCTION_LIST_DIR
project(skymarshal)

add_library(skymarshal_core INTERFACE include/lcm/lcm_coretypes.h)
target_include_directories(skymarshal_core INTERFACE include)
