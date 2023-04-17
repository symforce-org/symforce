/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <cstdint>
#include <limits>
#include <ostream>

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
  using subscript_t = std::int64_t;
  using superscript_t = std::int64_t;

  static constexpr char kInvalidLetter = static_cast<char>(0);
  static constexpr subscript_t kInvalidSub = std::numeric_limits<subscript_t>::min();
  static constexpr superscript_t kInvalidSuper = std::numeric_limits<superscript_t>::min();

  constexpr Key() = default;
  constexpr Key(char letter, subscript_t sub = kInvalidSub, superscript_t super = kInvalidSuper)
      : letter_(letter), sub_(sub), super_(super) {
    SYM_ASSERT(letter != kInvalidLetter);
  }

  constexpr Key(const key_t& key) : Key(key.letter, key.subscript, key.superscript) {}

  constexpr char Letter() const noexcept {
    return letter_;
  }

  constexpr subscript_t Sub() const noexcept {
    return sub_;
  }

  constexpr superscript_t Super() const noexcept {
    return super_;
  }

  key_t GetLcmType() const noexcept {
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

  constexpr bool operator==(const Key& other) const noexcept {
    return (other.letter_ == letter_) && (other.sub_ == sub_) && (other.super_ == super_);
  }

  constexpr bool operator!=(const Key& other) const noexcept {
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
  char letter_{kInvalidLetter};
  subscript_t sub_{kInvalidSub};
  superscript_t super_{kInvalidSuper};
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
