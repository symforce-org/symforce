# Compiling SymForce to Web Assembly

The goal is to create Javascript (ideally Typescript) bindings
for select C++ implementations within [SymForce](https://symforce.org/)
using [Embind](https://emscripten.org/docs/porting/connecting_cpp_and_javascript/embind.html)
so that symforce can be imported and used in JS like any other package. The strategy is similar to how we use pybind for Python.

Step 1: Source the emscripten SDK:
```
source ~/projects/wasymforce/.venv/bin/activate
source ~/projects/emsdk/emsdk_env.sh
```

Step 2: Run emcmake:
```
mkdir build_wasm
cd build_wasm

emcmake cmake .. \
-DSYMFORCE_BUILD_STATIC_LIBRARIES=ON \
-DSYMFORCE_BUILD_OPT=ON \
-DSYMFORCE_BUILD_CC_SYM=OFF \
-DSYMFORCE_BUILD_EXAMPLES=OFF \
-DSYMFORCE_BUILD_TESTS=OFF \
-DSYMFORCE_BUILD_SYMENGINE=OFF \
-DSYMFORCE_GENERATE_MANIFEST=OFF \
-DSYMFORCE_BUILD_BENCHMARKS=OFF
```

Step 3: Run emmake:
```
emmake make -j 7
```

Step 4: Run embind:
```
emcc -lembind -o symforce_bindings.js ../symforce/wasm/wasm_bindings.cc \
-s LLD_REPORT_UNDEFINED --no-entry \
-Wl,--whole-archive \
./_deps/spdlog-build/libspdlog.a \
./_deps/metis-build/libmetis/libmetis.a \
./_deps/fmtlib-build/libfmt.a \
./libsymforce_gen.a \
./symforce/opt/libsymforce_opt.a \
-Wl,--no-whole-archive \
-I ../gen/cpp \
-I ./_deps/eigen3-src \
-I ./lcmtypes/cpp \
-I ../third_party/skymarshal/include
```

