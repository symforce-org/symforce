function(get_package_and_type package_and_type package_outvar type_outvar)
  string(REPLACE " " ";" package_and_type ${package_and_type})
  list(GET package_and_type 0 package)
  list(GET package_and_type 1 type)
  set(${package_outvar} ${package} PARENT_SCOPE)
  set(${type_outvar} ${type} PARENT_SCOPE)
endfunction()

function(add_python_bindings
  target_name
  bindings_dir
  generated_files_outvar
  skymarshal_args_outvar
)
  set(types_to_generate ${ARGN})

  # NOTE(aaron): This is technically missing ${bindings_dir}/python2.7/lcmtypes/__init__.py,
  # which is also generated.  However, it's generated multiple times for multiple packages, and
  # cmake complains about the same output being in multiple rules or something.  I think we could
  # get around this by adding a target just for that, but it shouldn't _really_ be a problem...
  foreach(package_and_type ${types_to_generate})
    get_package_and_type(${package_and_type} package type)
    list(APPEND generated_files ${bindings_dir}/python2.7/lcmtypes/${package}/_${type}.py)
    list(APPEND generated_files ${bindings_dir}/python2.7/lcmtypes/${package}/__init__.py)
  endforeach()
  list(REMOVE_DUPLICATES generated_files)

  add_custom_target(${target_name}_py DEPENDS ${generated_files})

  set(${generated_files_outvar} ${generated_files} PARENT_SCOPE)

  list(APPEND skymarshal_args
    --python
    --python-path
    "${bindings_dir}/python2.7/lcmtypes"
    --python-namespace-packages
    --python-package-prefix
    lcmtypes
  )

  set(${skymarshal_args_outvar} ${skymarshal_args} PARENT_SCOPE)
endfunction()

function(add_cpp_bindings
  target_name
  bindings_dir
  generated_files_outvar
  skymarshal_args_outvar
)
  set(types_to_generate ${ARGN})

  foreach(package_and_type ${types_to_generate})
    get_package_and_type(${package_and_type} package type)
    list(APPEND generated_files ${bindings_dir}/cpp/lcmtypes/${package}/${type}.hpp)
  endforeach()

  add_library(${target_name}_cpp INTERFACE ${generated_files})
  target_include_directories(${target_name}_cpp INTERFACE ${bindings_dir}/cpp ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm)
  target_link_libraries(${target_name}_cpp INTERFACE skymarshal_core)

  set(${generated_files_outvar} ${generated_files} PARENT_SCOPE)

  list(APPEND skymarshal_args
    --cpp
    --cpp-hpath
    "${bindings_dir}/cpp/lcmtypes"
    --cpp-include
    lcmtypes
  )

  set(${skymarshal_args_outvar} ${skymarshal_args} PARENT_SCOPE)
endfunction()

function(add_skymarshal_bindings target_name bindings_dir lcmtypes_dir)
  if(NOT TARGET skymarshal_core)
    add_subdirectory(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.. EXCLUDE_FROM_ALL)
  endif()

  cmake_parse_arguments(args "" "" "LANGUAGES" ${ARGN})
  if(NOT args_LANGUAGES)
    set(args_LANGUAGES python cpp)
  endif()

  set(SKYMARSHAL_PYTHON env PYTHONPATH=${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../.. ${SYMFORCE_PYTHON})
  execute_process(
    COMMAND ${SKYMARSHAL_PYTHON} ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/parse_types.py ${lcmtypes_dir}
    OUTPUT_VARIABLE TYPES_TO_GENERATE
  )
  string(REPLACE "\n" ";" TYPES_TO_GENERATE_LIST ${TYPES_TO_GENERATE})

  foreach(language ${args_LANGUAGES})
    if(language STREQUAL "python")
      add_python_bindings(${target_name} ${bindings_dir} py_generated_files py_skymarshal_args ${TYPES_TO_GENERATE_LIST})
      list(APPEND skymarshal_args ${py_skymarshal_args})
      list(APPEND outputs ${py_generated_files})
    elseif(language STREQUAL "cpp")
      add_cpp_bindings(${target_name} ${bindings_dir} cpp_generated_files cpp_skymarshal_args ${TYPES_TO_GENERATE_LIST})
      list(APPEND skymarshal_args ${cpp_skymarshal_args})
      list(APPEND outputs ${cpp_generated_files})
    else()
      message(FATAL "Invalid language: ${language}")
    endif()
  endforeach()

  file(GLOB lcm_sources CONFIGURE_DEPENDS ${lcmtypes_dir}/*.lcm)
  add_custom_command(
    OUTPUT ${outputs}
    COMMAND ${SKYMARSHAL_PYTHON} -m skymarshal ${skymarshal_args} ${lcmtypes_dir}
    DEPENDS ${lcm_sources}
  )
endfunction()
