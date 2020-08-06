# For compiling codegen tests

***REMOVED***=***REMOVED***
export PKG_CONFIG_PATH=***REMOVED***/lib/pkgconfig/

INCLUDES = -I../gen/cpp -I./ -I***REMOVED***/include/eigen3
CXXFLAGS = -std=c++17 -L./ `pkg-config --cflags lcm`
LDFLAGS = -shared -fpic

CC_FILES =../gen/cpp/geo/*.cc

codegen_function_test: libsymforce_geo.so codegen_function_test.cc
	c++ $(CXXFLAGS) -o codegen_function_test codegen_function_test.cc -Wl,-rpath,'$$ORIGIN' -lsymforce_geo $(INCLUDES)

codegen_nan_test: libsymforce_geo.so codegen_nan_test.cc
	c++ $(CXXFLAGS) -o codegen_nan_test codegen_nan_test.cc -Wl,-rpath,'$$ORIGIN' -lsymforce_geo $(INCLUDES)

codegen_multi_function_test: libsymforce_geo.so codegen_multi_function_test.cc
	c++ $(CXXFLAGS) -o codegen_multi_function_test codegen_multi_function_test.cc -Wl,-rpath,'$$ORIGIN' -lsymforce_geo $(INCLUDES) -I./symforce_function_codegen_test_data/codegen_multi_function_data

libsymforce_geo.so: $(CC_FILES)
	time c++ $(CXXFLAGS) $(LDFLAGS) -o libsymforce_geo.so $(CC_FILES) $(INCLUDES)
