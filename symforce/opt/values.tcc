/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <stdexcept>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include "./fmt_compat.h"

#include "./assert.h"
#include "./values.h"

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
        fmt::format("Mismatched types; index entry for key {} is type {}, T is {}", entry.key,
                    entry.type, type));
  }

  // By default, we allow types that have alignment requirements larger than sizeof(Scalar),
  // and extract these using FromStorage.  But, for SymForce-native types, this should only be the
  // case for certain aligned Eigen types.  In SymForce CI builds, we set this flag to check this.
#if SYM_VALUES_DISALLOW_FROM_STORAGE_NON_EIGEN_TYPES
  constexpr bool use_from_storage = kIsEigenType<T> && alignof(T) > sizeof(Scalar);
#else
  constexpr bool use_from_storage = alignof(T) > sizeof(Scalar);
#endif

  // Construct the object
  if constexpr (use_from_storage) {
    return sym::StorageOps<T>::FromStorage(data_.data() + entry.offset);
  } else {
    // NOTE(aaron): In order to reinterpret_cast the result, we need 1) its size to fit into the
    // amount of memory it's given in the data_ array, and 2) its alignment to be less than or equal
    // to the alignment of the Scalar type (which will be its alignment in the data_ array).
    // If you have a type that meets the alignment requirement, but the size requirement, you should
    // fix its StorageDim, which is why use_from_storage only considers the alignment.
    static_assert(sizeof(T) <= sym::StorageOps<T>::StorageDim() * sizeof(Scalar));
    static_assert(alignof(T) <= sizeof(Scalar));
    return *reinterpret_cast<const T*>(data_.data() + entry.offset);
  }
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
void Values<Scalar>::SetNew(const Key& key, T&& value) {
  const auto added = Set(key, std::forward<T>(value));
  if (!added) {
    throw std::runtime_error(fmt::format("Key {} already exists", key));
  }
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

namespace internal {
template <typename T, typename = void>
struct MaybeTangentDim {
  static constexpr int32_t value = -1;
};

// Pre-C++17 void_t implementation
// See https://en.cppreference.com/w/cpp/types/void_t
template <typename... Ts>
struct make_void {
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

template <typename T>
struct MaybeTangentDim<T, void_t<decltype(LieGroupOps<T>::TangentDim())>> {
  static constexpr int32_t value = LieGroupOps<T>::TangentDim();
};
}  // namespace internal

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
    entry.tangent_dim = internal::MaybeTangentDim<T>::value;

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
  SYM_ASSERT((entry.offset + entry.storage_dim <= static_cast<int>(data_.size())));
  StorageOps<T>::ToStorage(value, data_.data() + entry.offset);
}

/**
 * Polymorphic helper to compute local coordinates
 */
template <typename T, typename Scalar = typename sym::StorageOps<T>::Scalar>
void LocalCoordinatesHelper(const Scalar* const storage_this, const Scalar* const storage_others,
                            Scalar* const tangent_out, const Scalar epsilon,
                            const int32_t /* tangent_dim */) {
  const T t1 = sym::StorageOps<T>::FromStorage(storage_this);
  const T t2 = sym::StorageOps<T>::FromStorage(storage_others);
  Eigen::Map<typename sym::LieGroupOps<T>::TangentVec> tangent_map(tangent_out);
  tangent_map = sym::LieGroupOps<T>::LocalCoordinates(t1, t2, epsilon);
}

template <typename Scalar>
void MatrixLocalCoordinatesHelper(const Scalar* const storage_this,
                                  const Scalar* const storage_others, Scalar* const tangent_out,
                                  const Scalar /* epsilon */, const int32_t tangent_dim) {
  for (int32_t i = 0; i < tangent_dim; ++i) {
    tangent_out[i] = storage_others[i] - storage_this[i];
  }
}

BY_TYPE_HELPER(LocalCoordinatesByType, LocalCoordinatesHelper, MatrixLocalCoordinatesHelper);

template <typename Scalar>
template <typename R, typename>
VectorX<Scalar> Values<Scalar>::LocalCoordinates(
    const Values<Scalar>& others, const R& keys, const Scalar epsilon,
    const std::optional<size_t> tangent_dimension) const {
  // If not provided, we must compute the total tangent dimension.
  const size_t tangent_dimension_final = [&]() {
    if (tangent_dimension) {
      return *tangent_dimension;
    }
    size_t tangent_dim = 0;
    for (const sym::Key& key : keys) {
      tangent_dim += map_.at(key).tangent_dim;
    }
    return tangent_dim;
  }();

  VectorX<Scalar> tangent_vec(tangent_dimension_final);
  size_t tangent_inx = 0;

  for (const sym::Key& key : keys) {
    const index_entry_t index_entry = map_.at(key);
    const index_entry_t other_index_entry = others.IndexEntryAt(key);
    LocalCoordinatesByType<Scalar>(index_entry.type, data_.data() + index_entry.offset,
                                   others.data_.data() + other_index_entry.offset,
                                   tangent_vec.data() + tangent_inx, epsilon,
                                   index_entry.tangent_dim);
    tangent_inx += index_entry.tangent_dim;
  }

  return tangent_vec;
}

}  // namespace sym
