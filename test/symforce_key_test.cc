/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <sstream>

#include <catch2/catch_test_macros.hpp>

#include <symforce/opt/key.h>

TEST_CASE("Key prints correctly", "[key]") {
  const sym::Key key1('l', -1820881232627931286, 209745344626);
  const sym::Key key2('l', 209745344626, -1947893602558886325);

  std::ostringstream ss;
  ss << key1 << "\n" << key2 << "\n";
  CHECK(ss.str() == "l_n1820881232627931286_209745344626\nl_209745344626_n1947893602558886325\n");
}

TEST_CASE("Key is initialized with default constructor", "[key]") {
  const sym::Key key{};

  CHECK(key.Letter() == sym::Key::kInvalidLetter);
  CHECK(key.Sub() == sym::Key::kInvalidSub);
  CHECK(key.Super() == sym::Key::kInvalidSuper);
}
