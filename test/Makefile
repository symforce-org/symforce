# For compiling codegen tests

INCLUDES = -I../gen/cpp -I./ -I***REMOVED***/include/eigen3
CXXFLAGS = -std=c++17 -L./ -O0
LDFLAGS = -shared -fpic

CC_FILES =../gen/cpp/geo/*.cc

codegen_function_test: libsymforce_geo.so codegen_function_test.cc
	c++ $(CXXFLAGS) -o codegen_function_test codegen_function_test.cc -Wl,-rpath,'$$ORIGIN' -lsymforce_geo $(INCLUDES)

libsymforce_geo.so: $(CC_FILES)
	time c++ $(CXXFLAGS) $(LDFLAGS) -o libsymforce_geo.so $(CC_FILES) $(INCLUDES)
