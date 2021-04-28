#include "./assert.h"

/**
 * Template method implementations for Values.
 */
namespace sym {

template <typename Scalar>
template <typename T>
T Values<Scalar>::At(const index_entry_t& entry) const {
  // Check the type
  // TODO(hayk): Depend on fmtlib?
  const type_t type = GetType<Scalar, T>();
  if (entry.type != type) {
    throw std::runtime_error("Mismatched types.");
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
  return At<T>(map_.at(key));
}

template <typename Scalar>
template <typename T>
bool Values<Scalar>::Set(const Key& key, const T& value) {
  static_assert(std::is_same<Scalar, typename sym::StorageOps<T>::Scalar>::value,
                "Calling Values.Set on mismatched scalar type.");
  const type_t type = GetType<Scalar, T>();
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
void Values<Scalar>::Set(const index_entry_t& entry, const T& value) {
  static_assert(std::is_same<Scalar, typename sym::StorageOps<T>::Scalar>::value,
                "Calling Values.Set on mismatched scalar type.");
  SYM_ASSERT((entry.type == GetType<Scalar, T>()));
  SYM_ASSERT((entry.offset + entry.storage_dim < data_.size()));
  sym::StorageOps<T>::ToStorage(value, data_.data() + entry.offset);
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
