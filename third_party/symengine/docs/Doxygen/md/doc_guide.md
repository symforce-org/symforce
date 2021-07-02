# Doxygen Style Guide

We write all the API documentation with [Doxygen](https://doxygen.nl). The
`add.h` and `add.cpp` are reference implementations of the style guide.

## Language and Locale

- Documentation should be in English.
- All dates are `YYYY-MM-DD`.
- We strongly discourage the use of Unicode symbols.
  - Symbols inside TeX math modes are fine.
- Numbers which are return values should not be in math mode.
- We use `snake_case` for the variables and functions.
- Use `.` at the end of each list item
- When adding in-text references, any style (APA, MLA, Chicago, IEEE, etc.) is allowed.
  - It is a good practice to add to the relevant `.bib` file in the repository.

### Markup

- Doxygen markup is preferred for single words.
  - `@a` for italics (parameters).
  - `@b` for bold (single emphasis).
  - `@c` for monospace (for file names / code symbols).
- For multiple words, we use the markdown syntax.
  - Keep the [Doxygen eccentricities](https://www.doxygen.nl/manual/markdown.html#markdown_dox) in mind.
  - We use the `#` style for headings.
- Never use HTML.

## Comments

- We use `@` for all [Doxygen special commands](https://www.doxygen.nl/manual/commands.html).

```unparsed
// Do
//!< @note

// ERROR -- DO NOT DO
//!< \note
```

- We use `//!<` for inline documentation (see above) so keep the [caveats in mind](https://www.doxygen.nl/manual/docblocks.html#memberdoc) from the Doxygen manual.

- Non documenting `//` and `/**/` are used to describe steps of an algorithm in-place.
  - We use [INLINE_SOURCES](https://www.doxygen.nl/manual/config.html#cfg_inline_sources) so they will show up in the documentation.

## Documentation Blocks

- We use the `/**` style.

```unparsed
/**
 *  @note This is acceptable
 */
```

## Indentation

- The author of the documentation is supposed to ensure that the documentation is legible in the code.
- We use one space before, and _two spaces_ after `*`.

```unparsed
/**
 *  @note This is acceptable since it has two spaces between the * and the word
 */
```

- We use one space to indicate a paragraph.

```unparsed
/**
 *  @note This is acceptable since it has two spaces between the * and the word.
 *   It is also an acceptable paragraph because of the leading indentation.
 */
```

```unparsed
/**
 *  @warning This is a pointlessly long paragraph to demonstrate the @b wrong
 *  way to indent a paragraph
 */
```

- Lists are to be indented for legibility as well.

```unparsed
/**
 *  @note I am a good list:
 *    - Item 1.
 *    - Item 2.
 *      - SubItem 1.
 *      - SubItem 2.
 *    - Item 3.
 *      - SubItem 1.
 */
```

## Source Files

Each file shall begin with a `@file` block as shown:

```unparsed
/**
 *  @file   blah.extension
 *  @author SymEngine Developers
 *  @date   2021-02-25
 *  @brief  A very brief but compelling explanation of this file
 *
 *  Created on: 2012-07-11
 *
 *  This is a single paragraph which really makes you feel like reading the rest
 *   of this file because it is really very interesting and has a bunch of @ref
 *   tags to make things easier
 */
```

The salient points are:

- `@file` must be present and is simply the filename.
- `@author` is **always** SymEngine Developers.
  - We let `git` handle more granular attribution.
- `@date` is the date the file was last modified.
  - This is inclusive of documentation changes.
- `@brief` should be a single line about the contents of the file.
- `Created on:` is not a Doxygen directive, but should be present.
- The paragraph is meant to describe the logical layout of the file.
  - `@ref` tags are meant to allow the user to jump to relevant sections more easily.

Note that this block is to appear before any `#ifndef` and `#include`
preprocessor directives.

#### Groups

It is in the `.h` file that `@ingroup` and `@defgroup` directives are used for
grouping. The logical grouping follows the layout of the `tests` directory.

### Headers

To maintain the logical division of `.cpp` and `.h` files, we disallow long comments in header files. This is also to reduce the compilation time when changing minor documentation[^1]. Anything longer than one line should be replaced with a short description which is expanded on in a `@note` in the corresponding `.cpp`. Acceptable special directives are (in order):

- `@brief` for a pithy description of the entity.
- `@pre` for describing preconditions (_optional_).
- `@see` for related functions.
- `@param` one for each input parameter.
  - `@param[out]` is not used, we have the `.cpp` to describe the effects of `void` functions.
- `@return` one for each possible return value with a line on the condition.
  - Keep this short, the `.cpp` has the implementation details.
  - `void` functions should `@return Void.` (including the `.`).
- `@relatesalso` takes a single class and groups the function with the class in the output (_optional_).
  - This is preceded by a line.

Note that since C++ is strongly typed, there is no need to describe the type of inputs (`@param`) or the outputs.

For example:

```unparsed
/**
 *  @brief The best number generator in the universe
 *  @pre Does assume a known universe
 *  @see `other_best` for an example of another number generator
 *  @param seed is the reproducibility helper
 *  @param am_awesome is an indicator of awesomeness
 *  @return the best number if the caller is awesome
 *  @return a random number if the caller is not awesome
 *  int best_number(const int &seed, std::string am_awesome);
*/
```

#### Inline Functions

These are typically simply a `@return` directive in a single line `//!<`; but should be used sparingly. If the function in question requires more documentation it probably should not be `inline`.

### Code Files

- The order of function definition must match the declaration in the header file

Here we expect:

- `@details` for explanations of the overall algorithm, speed concerns, etc.
- `@note` for longer details of the parameters or other important issues.
- `@warning` for pitfalls.
- Other special directives as required.
- `@see` is allowed in both headers and code files, but only sparingly.

```unparsed
/**
 *  @details The best number is determined by a complicated algorithm described
 *   over many paragraphs with equations here
 *  @f$ \mathbb{N}^3\to\mathbb{R}^N @f$
 */
int best_number(const int &seed, std::string am_awesome){
    return 0; // Clearly the documentation is lying to you
}
```

This formulation; with comments in the code, to augment the documentation blocks
is the most legible method.

## Source Code

### Classes

- We **do** expect private variables to be documented.

Classes descriptions have the `@class` directive declared before `@brief` in `.h` files and
before `@details` in the `.cpp` as well.

### Functions

- All documentation blocks must precede the function being documented.

## External Tools

We expect the build system to be able to find `graphviz` for `dot`.

[^1]: Changing a header triggers the recompilation of the entire project.
