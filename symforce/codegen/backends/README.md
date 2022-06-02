# Code Generation Backends

SymForce takes symbolic functions and generates runtime functions for multiple target backends. It aims to make it very straightforward to add new backends.

The minimal steps to support a new backend are:

 1. Choose a name for your backend (for example 'julia') and create a corresponding package in `symforce/codegen/backends`.
 2. Implement a subclass of SymPy `CodePrinter` that emits backend math code while traversing symbolic expressions. Sometimes SymPy already contains the backend and the best pattern is to inherit from it and customize as needed. The best way to do this is by looking at existing backends as examples.
 3. Implement a subclass of `symforce.codegen.codegen_config.CodegenConfig`. This is the spec that users pass to the `Codegen` object to use your backend. Again, see existing examples. Optionally import your config in `symforce/codegen/__init__.py`.
 4. Create a `templates` directory containing jinja templates that are used to generate the actual output files. They specify the high level structure and APIs around the math code. Your codegen config has a `templates_to_render` method that should match your templates. A typical start is just one function template.
 5. Add your backend's extensions to `FileType` in `symforce/codegen/template_util.py`, filling out relevant methods there.
 6. Add tests to `test/symforce_codegen_test.py`.

This will result in being able to generate functions for your backend that deal with scalars and arrays, but the `sym` geometry and camera classes. To implement those, follow the C++ and Python examples.
