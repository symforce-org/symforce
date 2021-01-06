# For compiling codegen tests
#
# Expects SYMFORCE_TEST_BACKEND to be set to either 'symengine' or 'sympy'

***REMOVED***=***REMOVED***
export PKG_CONFIG_PATH=***REMOVED***/lib/pkgconfig/

INCLUDES = -I../gen/cpp -I./ -I***REMOVED***/include/eigen3
CXXFLAGS = -std=c++17 -L./ `pkg-config --cflags lcm`
LDFLAGS = -shared -fpic

CC_FILES =../gen/cpp/geo/*.cc ../gen/cpp/geo/ops/**/*.cc

CC_FILES_VALUES =../symforce/opt/*.cc

codegen_%_test: codegen_%_test.cc libsymforce_geo.so
	c++ $(CXXFLAGS) -o codegen_$*_test codegen_$*_test.cc \
		-Wl,-rpath,'$$ORIGIN' -lsymforce_geo $(INCLUDES) \
		-I./symforce_function_codegen_test_data/$(SYMFORCE_TEST_BACKEND)/codegen_$*_test_data/cpp

libsymforce_geo.so: $(CC_FILES)
	time c++ $(CXXFLAGS) $(LDFLAGS) -o libsymforce_geo.so $(CC_FILES) $(INCLUDES)
