pybind11_add_module(
  cc_sym
  SHARED
  cc_factor.cc
  cc_key.cc
  cc_linearization.cc
  cc_optimization_stats.cc
  cc_optimizer.cc
  cc_sym.cc
  cc_values.cc
)

if(SYMFORCE_PY_EXTENSION_MODULE_OUTPUT_PATH)
  set_target_properties(cc_sym PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${SYMFORCE_PY_EXTENSION_MODULE_OUTPUT_PATH}
  )
endif()

target_link_libraries(
  cc_sym
  PRIVATE
  symforce_opt
  symforce_gen
)
