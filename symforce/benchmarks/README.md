# Benchmarks

This package contains benchmarks for SymForce and comparative examples to other libraries.

First install these system deps in addition to the core symforce ones from the main README:

```
conda install boost
conda install -c conda-forge clang
```

To build benchmark examples and get meaningful results, run cmake with:

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-march=native -ffast-math" -DSYMFORCE_BUILD_BENCHMARKS=ON
```

You also need to make sure `perf` is installed.

You can run benchmark examples and save timing info with `python benchmarks/run_benchmarks.py`.
