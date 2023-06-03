/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

/** @file
 * Utility for timing execution of various scopes in a program.  Timings are aggregated across all
 * runs of each scope in an execution, and optionally printed in a table on program exit.  Example
 * usage:
 *
 *     void Foo() {
 *         SYM_TIME_SCOPE("Foo");
 *         do_stuff();
 *
 *         {
 *             SYM_TIME_SCOPE("Foo::more_stuff");
 *             do_more_stuff();
 *         }
 *     }
 *
 * SymForce has a default implementation of this timing and aggregation mechanism; if you have some
 * other timing system that you'd like SymForce to hook into, you can define a header to include
 * with SYMFORCE_TIC_TOC_HEADER and provide your own definition of the SYM_TIME_SCOPE macro
 */

#ifdef SYMFORCE_TIC_TOC_HEADER
#include SYMFORCE_TIC_TOC_HEADER

#ifndef SYM_TIME_SCOPE
#error The SYM_TIME_SCOPE macro must be defined if SYMFORCE_TIC_TOC_HEADER is provided
#endif

#else  // if !defined(SYMFORCE_TIC_TOC_HEADER)
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/fmt/bundled/ostream.h>

#include "./internal/tic_toc.h"

#ifndef SYM_TIME_SCOPE
#define _SYMFORCE_OPT_INTERNAL_COMBINE1(X, Y) X##Y
#define _SYMFORCE_OPT_INTERNAL_COMBINE(X, Y) _SYMFORCE_OPT_INTERNAL_COMBINE1(X, Y)
#define SYM_TIME_SCOPE(fmt_str, ...)                          \
  sym::internal::ScopedTicToc _SYMFORCE_OPT_INTERNAL_COMBINE( \
      scope_timer_, __LINE__)((fmt::format(fmt_str, ##__VA_ARGS__)))
#endif

#endif  // defined(SYMFORCE_TIC_TOC_HEADER)
