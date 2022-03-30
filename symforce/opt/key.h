/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <limits>
#include <ostream>
#include <stdint.h>

#include <lcmtypes/sym/key_t.hpp>

#include "./assert.h"

namespace sym {

/**
 * Key type for Values. Contains a letter plus an integral subscript and superscript.
 * Can construct with a letter, a letter + sub, or a letter + sub + super, but not letter + super.
 *
 * TODO(hayk): Consider an abstraction where Key contains a type enum.
 */
class Key {
 public:
  using subscript_t = int64_t;
  using superscript_t = int64_t;

  static constexpr char kInvalidLetter = static_cast<char>(0);
  static constexpr subscript_t kInvalidSub = std::numeric_limits<subscript_t>::min();
  static constexpr superscript_t kInvalidSuper = std::numeric_limits<superscript_t>::min();

  Key(const char letter, const subscript_t sub, const superscript_t super)
      : letter_(letter), sub_(sub), super_(super) {
    SYM_ASSERT(letter != kInvalidLetter);
  }
  Key(const char letter, const subscript_t sub) : Key(letter, sub, kInvalidSuper) {}
  Key(const char letter) : Key(letter, kInvalidSub, kInvalidSuper) {}

  Key(const key_t& key) : Key(key.letter, key.subscript, key.superscript) {}

  Key() {}

  char Letter() const {
    return letter_;
  }

  subscript_t Sub() const {
    return sub_;
  }

  superscript_t Super() const {
    return super_;
  }

  key_t GetLcmType() const {
    key_t key;
    key.letter = letter_;
    key.subscript = sub_;
    key.superscript = super_;
    return key;
  }

  /**
   * Create a new Key from an existing Key and a superscript.  The superscript on the existing Key
   * must be empty
   */
  static Key WithSuper(const Key& key, const superscript_t super) {
    SYM_ASSERT(key.Super() == kInvalidSuper);
    return Key(key.Letter(), key.Sub(), super);
  }

  bool operator==(const Key& other) const {
    return (other.letter_ == letter_) && (other.sub_ == sub_) && (other.super_ == super_);
  }

  bool operator!=(const Key& other) const {
    return !(*this == other);
  }

  /**
   * Return true if a is LESS than b, in dictionary order of the tuple (letter, sub, super).
   */
  static bool LexicalLessThan(const Key& a, const Key& b);

  /**
   * Implementation of the Compare spec for use in containers
   */
  struct LexicalCompare {
    bool operator()(const Key& a, const Key& b) const {
      return LexicalLessThan(a, b);
    }
  };

 protected:
  char letter_;
  subscript_t sub_;
  superscript_t super_;
};

/**
 * Print implementation for Key.
 *
 * Examples
 *     Key('C', 13) -> "C_13"
 *     Key('f') -> "f"
 *     Key('f', 32, 2) -> "f_32_2"
 *     Key('A', -2, 123) -> "A_n2_123"
 *     Key() -> "NULLKEY"
 */
std::ostream& operator<<(std::ostream& os, const sym::Key& key);
std::ostream& operator<<(std::ostream& os, const sym::key_t& key);

}  // namespace sym

/**
 * Hash function for Key.
 */
namespace std {

template <>
struct hash<sym::Key> {
  std::size_t operator()(const sym::Key& key) const;
};

template <>
struct hash<sym::key_t> {
  std::size_t operator()(const sym::key_t& key) const;
};

}  // namespace std
