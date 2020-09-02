#include "./key.h"

std::ostream& operator<<(std::ostream& os, const sym::Key& key) {
  if (key.Letter() == sym::Key::kInvalidLetter) {
    os << "NULLKEY";
    return os;
  }

  os << key.Letter();

  if (key.Sub() != sym::Key::kInvalidSub) {
    os << '_';
    if (key.Sub() < 0) {
      os << 'n';
    }
    os << abs(key.Sub());
  }

  if (key.Super() != sym::Key::kInvalidSuper) {
    os << '_';
    if (key.Super() < 0) {
      os << 'n';
    }
    os << abs(key.Super());
  }

  return os;
}

std::ostream& operator<<(std::ostream& os, const sym::key_t& key) {
  os << sym::Key(key);
  return os;
}
