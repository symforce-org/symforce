/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <fmt/format.h>
#include <fmt/ostream.h>

#include "./assert.h"

/**
 * Template method implementations for Values.
 */
namespace sym {

// ----------------------------------------------------------------------------
// Public Methods
// ----------------------------------------------------------------------------

template <typename Scalar>
template <typename T>
T Values<Scalar>::At(const index_entry_t& entry) const {
  // Check the type
  const type_t type = StorageOps<T>::TypeEnum();
  if (entry.type != type) {
    throw std::runtime_error(
        fmt::format("Mismatched types; index entry is type {}, T is {}", entry.type, type));
  }

  // Construct the object
#if 1
  return sym::StorageOps<T>::FromStorage(data_.data() + entry.offset);
#else
  // NOTE(hayk): It could be more efficient to reinterpret_cast here, and we could provide
  // mutable references if desired. But also technically undefined?
  return *reinterpret_cast<const T*>(data_.data() + entry.offset);
#endif
}

template <typename Scalar>
template <typename T>
T Values<Scalar>::At(const Key& key) const {
  return At<T>(IndexEntryAt(key));
}

template <typename Scalar>
template <typename T>
std::enable_if_t<!kIsEigenType<T>, bool> Values<Scalar>::Set(const Key& key, const T& value) {
  return SetInternal<T>(key, value);
}

template <typename Scalar>
template <typename Derived>
std::enable_if_t<kIsEigenType<Derived>, bool> Values<Scalar>::Set(const Key& key,
                                                                  const Derived& value) {
  return SetInternal<typename Derived::PlainMatrix>(key, value);
}

template <typename Scalar>
template <typename T>
std::enable_if_t<!kIsEigenType<T>> Values<Scalar>::Set(const index_entry_t& entry, const T& value) {
  SetInternal<T>(entry, value);
}

template <typename Scalar>
template <typename Derived>
std::enable_if_t<kIsEigenType<Derived>> Values<Scalar>::Set(const index_entry_t& entry,
                                                            const Derived& value) {
  SetInternal<typename Derived::PlainMatrix>(entry, value);
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

template <typename Scalar>
template <typename T>
bool Values<Scalar>::SetInternal(const Key& key, const T& value) {
  static_assert(std::is_same<Scalar, typename StorageOps<T>::Scalar>::value,
                "Calling Values.Set on mismatched scalar type.");
  const type_t type = StorageOps<T>::TypeEnum();
  bool is_new = false;

  // Create the entry if not present.
  auto it = map_.find(key);
  if (it == map_.end()) {
    it = map_.emplace(key, index_entry_t{}).first;
  }
  index_entry_t& entry = it->second;

  if (entry.type == type_t::INVALID) {
    is_new = true;
    entry.key = key.GetLcmType();
    entry.type = type;
    entry.offset = static_cast<int32_t>(data_.size());
    entry.storage_dim = sym::StorageOps<T>::StorageDim();
    entry.tangent_dim = sym::LieGroupOps<T>::TangentDim();

    // Extend end of data
    data_.insert(data_.end(), entry.storage_dim, 0);
  } else {
    if (entry.type != type) {
      // TODO(hayk): Return an error enum instead of an exception?
      throw std::runtime_error("Calling Set on the wrong value type.");
    }
  }

  // Save the value
  sym::StorageOps<T>::ToStorage(value, data_.data() + entry.offset);
  return is_new;
}

template <typename Scalar>
template <typename T>
void Values<Scalar>::SetInternal(const index_entry_t& entry, const T& value) {
  static_assert(std::is_same<Scalar, typename StorageOps<T>::Scalar>::value,
                "Calling Values.Set on mismatched scalar type.");
  SYM_ASSERT((entry.type == StorageOps<T>::TypeEnum()));
  SYM_ASSERT((entry.offset + entry.storage_dim <= data_.size()));
  StorageOps<T>::ToStorage(value, data_.data() + entry.offset);
}

// ----------------------------------------------------------------------------
// LCM type alias
// ----------------------------------------------------------------------------

template <>
struct ValuesLcmTypeHelper<double> {
  using Type = values_t;
};

template <>
struct ValuesLcmTypeHelper<float> {
  using Type = valuesf_t;
};

}  // namespace sym
