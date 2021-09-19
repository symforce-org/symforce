# For compiling codegen tests
#
# Expects SYMFORCE_TEST_BACKEND to be set to either 'symengine' or 'sympy'

***REMOVED***=***REMOVED***
export PKG_CONFIG_PATH=***REMOVED***/lib/pkgconfig/

SPDLOG_INCLUDE_DIR=***REMOVED***/_deps/spdlog-src/include

INCLUDES = -I../gen/cpp \
	-I***REMOVED***/include/eigen3 \
	-I../third_party/catch2/include \
	-I$(***REMOVED***)/gen/lcmtypes/cpp
CXXFLAGS = -std=c++14 -L./ `pkg-config --cflags lcm`
LDFLAGS = -shared -fpic


CC_FILES =../gen/cpp/sym/*.cc ../gen/cpp/sym/ops/**/*.cc

CC_FILES_VALUES =../symforce/opt/*.cc

codegen_%_test: codegen_%_test.cc libsymforce_gen.so libcatch2.so
	c++ $(CXXFLAGS) -o codegen_$*_test codegen_$*_test.cc \
		-Wl,-rpath,'$$ORIGIN' -lsymforce_gen -lcatch2 $(INCLUDES) \
		-I$(SPDLOG_INCLUDE_DIR) \
		-I./symforce_function_codegen_test_data/$(SYMFORCE_TEST_BACKEND)/codegen_$*_test_data/cpp

libsymforce_gen.so: $(CC_FILES)
	time c++ $(CXXFLAGS) $(LDFLAGS) -o libsymforce_gen.so $(CC_FILES) $(INCLUDES)

libcatch2.so: ../third_party/catch2/src/catch_amalgamated.cpp catch_main.cc
	c++ $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(INCLUDES)
