# How to Contribute

Please report all bugs, issues, comments and feature requests into our
[issues](https://github.com/sympy/symengine/issues).

We welcome all Pull Requests, just send the code and we will help you improve
it. You can read the [Style Guide](docs/Doxygen/md/style_guide.md) and
[Design](doc/design.md) that your code should follow, but do not worry if you
miss anything, we will let you know in the review.

# Code Formatting

## CI

The code formatting of each PR is checked by the CI using clang-format.

If changes are required, a comment will be added to the PR with instructions for fixing the formatting.

You can also do this formatting locally (if you have clang-format installed) by running `./bin/test_format_local.sh`

It is preferable to use the same version of clang-format as the CI does (currently version 11), as different versions can give different results.

## Installing clang-format

- MacOS
  - `brew install clang-format`
  - https://formulae.brew.sh/formula/clang-format
- Debian/Ubuntu:
  - `sudo apt install clang-format-11`
  - see https://apt.llvm.org/ if not available in your version of Debian/Ubuntu
- Conda:
  - https://anaconda.org/conda-forge/clangdev

## Git hook

To add a git hook that formats the code locally on every commit, you can run this command from the root of the symengine repo:

```
ln $(pwd)/bin/test_format_local.sh .git/hooks/pre-commit
```

To remove the hook, remove the `.git/hooks/pre-commit` file.
