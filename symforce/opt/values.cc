/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./values.h"

#include <fmt/format.h>
#include <fmt/ostream.h>

#include "./assert.h"

namespace sym {

template <typename Scalar>
Values<Scalar>::Values(std::initializer_list<Values<Scalar>> others) {
  for (const auto& other : others) {
    // add in the Keys with a different offset
    const size_t offset = data_.size();
    for (const auto& it : other.map_) {
      SYM_ASSERT(map_.find(it.first) == map_.end());
      map_[it.first] = it.second;
      map_[it.first].offset += offset;
    }

    // copy data from other Values
    data_.insert(data_.end(), other.data_.begin(), other.data_.end());
  }
}

template <typename Scalar>
Values<Scalar>::Values(const LcmType& msg) {
  for (const index_entry_t& entry : msg.index.entries) {
    map_[entry.key] = entry;
  }
  data_ = msg.data;
}

template <typename Scalar>
bool Values<Scalar>::Has(const Key& key) const {
  return map_.find(key) != map_.end();
}

template <typename Scalar>
void Values<Scalar>::UpdateOrSet(const index_t& index, const Values<Scalar>& other) {
  for (const auto& entry_other : index.entries) {
    const auto offset_other = other.data_.begin() + entry_other.offset;
    const Key key(entry_other.key);
    auto it = map_.find(key);
    // insert keys if not existed
    if (it == map_.end()) {
      it = map_.emplace(key, index_entry_t{}).first;
      index_entry_t& entry_this = it->second;
      entry_this = entry_other;
      entry_this.offset = static_cast<int32_t>(data_.size());
      // extend end of data
      data_.insert(data_.end(), offset_other, offset_other + entry_other.storage_dim);
    } else {
      std::copy_n(offset_other, entry_other.storage_dim, data_.begin() + it->second.offset);
    }
  }
}

template <typename Scalar>
size_t Values<Scalar>::NumEntries() const {
  return map_.size();
}

template <typename Scalar>
std::vector<Key> Values<Scalar>::Keys(const bool sort_by_offset) const {
  std::vector<Key> keys;
  keys.reserve(map_.size());
  for (const auto& kv : map_) {
    keys.push_back(kv.first);
  }

  // Sort the keys by offset so iterating through is saner and more memory friendly
  if (sort_by_offset) {
    std::sort(keys.begin(), keys.end(), [&](const sym::Key& a, const sym::Key& b) {
      return map_.at(a).offset < map_.at(b).offset;
    });
  }

  return keys;
}

template <typename Scalar>
const typename Values<Scalar>::MapType& Values<Scalar>::Items() const {
  return map_;
}

template <typename Scalar>
const typename Values<Scalar>::ArrayType& Values<Scalar>::Data() const {
  return data_;
}

template <typename Scalar>
template <typename NewScalar>
Values<NewScalar> Values<Scalar>::Cast() const {
  Values<NewScalar> new_values{};
  new_values.map_ = map_;

  // This shouldn't really be less efficient in the Scalar == NewScalar case
  new_values.data_.resize(data_.size());
  std::copy(data_.begin(), data_.end(), new_values.data_.begin());

  return new_values;
}

template <typename Scalar>
bool Values<Scalar>::Remove(const Key& key) {
  size_t num_removed = map_.erase(key);
  return static_cast<bool>(num_removed);
}

template <typename Scalar>
void Values<Scalar>::RemoveAll() {
  map_.clear();
  data_.clear();
}

template <typename Scalar>
size_t Values<Scalar>::Cleanup() {
  // Copy the original data
  const ArrayType data_copy = data_;

  // Build an index of all keys
  const index_t full_index = CreateIndex(Keys());

  // Re-allocate data to the appropriate size
  data_.resize(full_index.storage_dim);
  SYM_ASSERT(data_copy.size() >= data_.size());

  // Copy into new data and update the offset in the map
  size_t new_offset = 0;
  for (const index_entry_t& entry : full_index.entries) {
    std::copy_n(data_copy.begin() + entry.offset, entry.storage_dim, data_.begin() + new_offset);
    map_[entry.key].offset = new_offset;
    new_offset += entry.storage_dim;
  }
  return data_copy.size() - data_.size();
}

template <typename Scalar>
index_t Values<Scalar>::CreateIndex(const std::vector<Key>& keys) const {
  index_t index{};
  index.entries.reserve(keys.size());
  for (const Key& key : keys) {
    const auto it = map_.find(key);

    if (it == map_.end()) {
      throw std::runtime_error(fmt::format("Tried to create index for key {} not in values", key));
    }

    const auto& entry = it->second;
    index.entries.push_back(entry);
    index.storage_dim += entry.storage_dim;
    index.tangent_dim += entry.tangent_dim;
  }
  return index;
}

template <typename Scalar>
index_entry_t Values<Scalar>::IndexEntryAt(const Key& key) const {
  const auto it = map_.find(key);
  if (it == map_.end()) {
    throw std::runtime_error(fmt::format("Key not found: {}", key));
  }
  return it->second;
}

template <typename Scalar>
void Values<Scalar>::FillLcmType(LcmType& msg, bool sort_keys) const {
  msg.index = CreateIndex(Keys(sort_keys));
  msg.data = data_;
}

template <typename Scalar>
void Values<Scalar>::FillLcmType(LcmType* msg, bool sort_keys) const {
  SYM_ASSERT(msg != nullptr);
  FillLcmType(*msg, sort_keys);
}

template <typename Scalar>
typename Values<Scalar>::LcmType Values<Scalar>::GetLcmType(bool sort_keys) const {
  LcmType msg;
  FillLcmType(msg, sort_keys);
  return msg;
}

template <typename Scalar>
void Values<Scalar>::Update(const index_t& index, const Values<Scalar>& other) {
  SYM_ASSERT(data_.size() == other.data_.size());
  for (const index_entry_t& entry : index.entries) {
    std::copy_n(other.data_.begin() + entry.offset, entry.storage_dim,
                data_.begin() + entry.offset);
  }
}

template <typename Scalar>
void Values<Scalar>::Update(const index_t& index_this, const index_t& index_other,
                            const Values<Scalar>& other) {
  SYM_ASSERT(index_this.entries.size() == index_other.entries.size());
  for (int i = 0; i < static_cast<int>(index_this.entries.size()); ++i) {
    const index_entry_t& entry_this = index_this.entries[i];
    const index_entry_t& entry_other = index_other.entries[i];
    SYM_ASSERT(entry_this.storage_dim == entry_other.storage_dim);
    SYM_ASSERT(entry_this.key == entry_other.key);
    std::copy_n(other.data_.begin() + entry_other.offset, entry_this.storage_dim,
                data_.begin() + entry_this.offset);
  }
}

/**
 * Polymorphic helper to apply a retraction.
 */
template <typename T, typename Scalar = typename sym::StorageOps<T>::Scalar>
void RetractHelper(const Scalar* tangent_data, const Scalar epsilon, Scalar* const t_ptr,
                   const int32_t /* tangent_dim */) {
  static_assert(!kIsEigenType<T>, "Eigen types not supported");

  const T t_in = sym::StorageOps<T>::FromStorage(t_ptr);
  const typename sym::LieGroupOps<T>::TangentVec tangent_vec(tangent_data);
  const T t_out = sym::LieGroupOps<T>::Retract(t_in, tangent_vec, epsilon);
  sym::StorageOps<T>::ToStorage(t_out, t_ptr);
}

template <typename Scalar>
void MatrixRetractHelper(const Scalar* tangent_data, const Scalar /* epsilon */,
                         Scalar* const t_ptr, const int32_t tangent_dim) {
  for (int32_t i = 0; i < tangent_dim; ++i) {
    t_ptr[i] += tangent_data[i];
  }
}
BY_TYPE_HELPER(RetractByType, RetractHelper, MatrixRetractHelper);

template <typename Scalar>
void Values<Scalar>::Retract(const index_t& index, const Scalar* delta, const Scalar epsilon) {
  size_t tangent_inx = 0;
  for (const index_entry_t& entry : index.entries) {
    RetractByType<Scalar>(entry.type, /* tangent_data */ delta + tangent_inx, epsilon,
                          /* t_ptr */ data_.data() + entry.offset, entry.tangent_dim);
    tangent_inx += entry.tangent_dim;
  }
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
  const typename sym::LieGroupOps<T>::TangentVec tangent_vec =
      sym::LieGroupOps<T>::LocalCoordinates(t2, t1, epsilon);
  // TODO(alvin): can we avoid this copy?
  std::copy_n(tangent_vec.data(), sym::LieGroupOps<T>::TangentDim(), tangent_out);
}
template <typename Scalar>
void MatrixLocalCoordinatesHelper(const Scalar* const storage_this,
                                  const Scalar* const storage_others, Scalar* const tangent_out,
                                  const Scalar /* epsilon */, const int32_t tangent_dim) {
  for (int32_t i = 0; i < tangent_dim; ++i) {
    tangent_out[i] = storage_this[i] - storage_others[i];
  }
}

BY_TYPE_HELPER(LocalCoordinatesByType, LocalCoordinatesHelper, MatrixLocalCoordinatesHelper);

template <typename Scalar>
VectorX<Scalar> Values<Scalar>::LocalCoordinates(const Values<Scalar>& others, const index_t& index,
                                                 const Scalar epsilon) {
  VectorX<Scalar> tangent_vec(index.tangent_dim);
  size_t tangent_inx = 0;

  for (const index_entry_t& entry : index.entries) {
    LocalCoordinatesByType<Scalar>(entry.type, data_.data() + entry.offset,
                                   others.data_.data() + entry.offset,
                                   tangent_vec.data() + tangent_inx, epsilon, entry.tangent_dim);
    tangent_inx += entry.tangent_dim;
  }

  return tangent_vec;
}

namespace {

template <typename T>
std::string FormatHelper(const type_t /* type */, const typename StorageOps<T>::Scalar* data_ptr,
                         const int32_t /* storage_dim */) {
  return fmt::format("{}", StorageOps<T>::FromStorage(data_ptr));
}
template <typename Scalar>
std::string MatrixFormatHelper(const type_t type, const Scalar* data_ptr,
                               const int32_t storage_dim) {
  return fmt::format("<{} {}>", type, Eigen::Map<const VectorX<Scalar>>(data_ptr, storage_dim));
}
BY_TYPE_HELPER(FormatByType, FormatHelper, MatrixFormatHelper);

}  // namespace

// ----------------------------------------------------------------------------
// Printing
// ----------------------------------------------------------------------------

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const Values<Scalar>& v) {
  // Make an index so we iterate through in offset order
  const index_t index = v.CreateIndex(v.Keys(/* sort by offset */ true));

  // Print header
  fmt::print(os, "<Values{} entries={} array={} storage_dim={} tangent_dim={}\n",
             typeid(Scalar).name(), index.entries.size(), v.Data().size(), index.storage_dim,
             index.tangent_dim);

  // Print each element
  for (const index_entry_t& entry : index.entries) {
    fmt::print(os, " {} [{}:{}] --> {}\n", Key(entry.key), entry.offset,
               entry.offset + entry.storage_dim,
               FormatByType<Scalar>(entry.type, entry.type, v.Data().data() + entry.offset,
                                    entry.storage_dim));
  }

  os << ">";
  return os;
}

template std::ostream& operator<< <float>(std::ostream& os, const Values<float>& v);
template std::ostream& operator<< <double>(std::ostream& os, const Values<double>& v);

}  // namespace sym

// Explicit instantiation
template class sym::Values<double>;
template class sym::Values<float>;

template sym::Values<double> sym::Values<double>::Cast<double>() const;
template sym::Values<float> sym::Values<double>::Cast<float>() const;
template sym::Values<double> sym::Values<float>::Cast<double>() const;
template sym::Values<float> sym::Values<float>::Cast<float>() const;
