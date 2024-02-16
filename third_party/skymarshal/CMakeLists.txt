cmake_minimum_required(VERSION 3.19)  # For interface library headers
project(skymarshal)

add_library(skymarshal_core INTERFACE include/lcm/lcm_coretypes.h include/lcm/lcm_json.hpp include/lcm/lcm_json_eigen.hpp)
target_include_directories(skymarshal_core SYSTEM INTERFACE include)
install(DIRECTORY include/ DESTINATION include)
