/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <unordered_map>

#include <lcmtypes/sym/values_t.hpp>
#include <lcmtypes/sym/valuesf_t.hpp>

#include <sym/util/type_ops.h>

#include "./key.h"

namespace sym {

template <typename _S>
struct ValuesLcmTypeHelper;

/**
 * Efficient polymorphic data structure to store named types with a dict-like interface and
 * support efficient repeated operations using a key index. Supports on-manifold optimization.
 *
 * Compatible types are given by the type_t enum. All types implement the StorageOps and
 * LieGroupOps concepts, which are the core operating mechanisms in this class.
 */
template <typename Scalar>
class Values {
 public:
  using MapType = std::unordered_map<Key, index_entry_t>;
  using ArrayType = std::vector<Scalar>;

  // Expose the correct LCM type (values_t or valuesf_t)
  using LcmType = typename ValuesLcmTypeHelper<Scalar>::Type;

  /**
   * Default construct as empty.
   */
  Values();

  /**
   * Construct from a list of other Values objects. The order of Keys are preserved by
   * the order of the Values in the initializer list
   *
   * NOTE(alvin): others Values should not contain overlapping Keys
   */
  explicit Values(std::initializer_list<Values<Scalar>> others);

  /**
   * Construct from serialized form.
   */
  explicit Values(const LcmType& msg);

  /**
   * Return whether the key exists.
   */
  bool Has(const Key& key) const;

  /**
   * Retrieve a value by key.
   */
  template <typename T>
  T At(const Key& key) const;

  /**
   * Add or update a value by key. Returns true if added, false if updated.
   *
   * Overload for non-Eigen types
   */
  template <typename T>
  std::enable_if_t<!kIsEigenType<T>, bool> Set(const Key& key, const T& value);

  /**
   * Add or update a value by key. Returns true if added, false if updated.
   *
   * Overload for Eigen types
   */
  template <typename Derived>
  std::enable_if_t<kIsEigenType<Derived>, bool> Set(const Key& key, const Derived& value);

  /**
   * Update or add keys to this Values base on other Values of different structure.
   * index MUST be valid for other.
   *
   * NOTE(alvin): it is less efficient than the Update methods below if index objects are
   *              created and cached. This method performs map lookup for each key of the index
   */
  void UpdateOrSet(const index_t& index, const Values<Scalar>& other);

  /**
   * Number of keys.
   */
  int32_t NumEntries() const;

  /**
   * Has zero keys.
   */
  bool Empty() const {
    return NumEntries() == 0;
  }

  /**
   * Get all keys.
   *
   * Args:
   *   sort_by_offset: Sorts by storage order to make iteration safer and more memory efficient
   *
   * NOTE(hayk): If we changed key storage to a sorted vector this could automatically be maintained
   * and it would be more embedded friendly, but At(key) would become O(N) for linear search.
   */
  std::vector<Key> Keys(const bool sort_by_offset = true) const;

  /**
   * Expose map type to allow iteration.
   */
  const MapType& Items() const;

  /**
   * Raw data buffer.
   */
  const ArrayType& Data() const;

  /**
   * Cast to another Scalar type (returns a copy)
   */
  template <typename NewScalar>
  Values<NewScalar> Cast() const;

  /**
   * Remove the given key. Only removes the index entry, does not change the data array.
   * Returns true if removed, false if already not present.
   *
   * Call Cleanup() to re-pack the data array.
   */
  bool Remove(const Key& key);

  /**
   * Remove all keys and empty out the storage.
   */
  void RemoveAll();

  /**
   * Repack the data array to get rid of empty space from removed keys. If regularly removing
   * keys, it's up to the user to call this appropriately to avoid storage growth. Returns the
   * number of Scalar elements cleaned up from the data array.
   *
   * It will INVALIDATE all indices, offset increments, and pointers.
   * Re-create an index with CreateIndex().
   */
  size_t Cleanup();

  /**
   * Create an index from the given ordered subset of keys. This object can then be used
   * for repeated efficient operations on that subset of keys.
   *
   * If you want an index of all the keys, call `values.CreateIndex(values.Keys())`.
   *
   * An index will be INVALIDATED if the following happens:
   *  1) Remove() is called with a contained key, or RemoveAll() is called
   *  2) Cleanup() is called to re-pack the data array
   *
   * NOTE(hayk): We could also add a simple UpdateIndex(&index) method, since the offset is the
   * only thing that needs to get updated after repacking.
   */
  index_t CreateIndex(const std::vector<Key>& keys) const;

  /**
   * Retrieve an index entry by key. This performs a map lookup.
   *
   * An index entry will be INVALIDATED if the following happens:
   *  1) Remove() is called with the indexed key, or RemoveAll() is called
   *  2) Cleanup() is called to re-pack the data array
   */
  index_entry_t IndexEntryAt(const Key& key) const;

  /**
   * Retrieve a value by index entry. This avoids a map lookup compared to At(key).
   */
  template <typename T>
  T At(const index_entry_t& entry) const;

  /**
   * Update a value by index entry with no map lookup (compared to Set(key)).
   * This does NOT add new values and assumes the key exists already.
   *
   * Overload for non-Eigen types
   */
  template <typename T>
  std::enable_if_t<!kIsEigenType<T>> Set(const index_entry_t& key, const T& value);

  /**
   * Update a value by index entry with no map lookup (compared to Set(key)).
   * This does NOT add new values and assumes the key exists already.
   *
   * Overload for Eigen types
   */
  template <typename Derived>
  std::enable_if_t<kIsEigenType<Derived>> Set(const index_entry_t& key, const Derived& value);

  /**
   * Efficiently update the keys given by this index from other into this. This purely copies
   * slices of the data arrays, the index MUST be valid for both objects!
   */
  void Update(const index_t& index, const Values<Scalar>& other);

  /**
   * Efficiently update the keys from a different structured Values, given by
   * this index and other index. This purely copies slices of the data arrays.
   * index_this MUST be valid for this object; index_other MUST be valid for other object.
   */
  void Update(const index_t& index_this, const index_t& index_other, const Values<Scalar>& other);

  /**
   * Perform a retraction from an update vector.
   *
   * Args:
   *   index: Ordered list of keys in the delta vector
   *   delta: Pointer to update vector - MUST be the size of index.tangent_dim!
   *   epsilon: Small constant to avoid singularities (do not use zero)
   */
  void Retract(const index_t& index, const Scalar* delta, const Scalar epsilon);

  /**
   * Express this Values in the local coordinate of others Values, i.e., this \ominus others
   *
   * Args:
   *   others: The other Values that the local coordinate is relative to
   *   index: Ordered list of keys to include (MUST be valid for both this and others Values)
   *   epsilon: Small constant to avoid singularities (do not use zero)
   */
  VectorX<Scalar> LocalCoordinates(const Values<Scalar>& others, const index_t& index,
                                   const Scalar epsilon);

  /**
   * Serialize to LCM.
   */
  void FillLcmType(LcmType* msg) const;
  LcmType GetLcmType() const;

 protected:
  MapType map_;
  ArrayType data_;

  template <typename T>
  bool SetInternal(const sym::Key& key, const T& value);

  template <typename T>
  void SetInternal(const index_entry_t& entry, const T& value);

  template <typename OtherScalar>
  friend class Values;
};

// Shorthand instantiations
using Valuesd = Values<double>;
using Valuesf = Values<float>;

/**
 * Prints entries with their keys, data slices, and values, like:
 *
 *   Valuesd entries=4 array=8 storage_dim=7 tangent_dim=6
 *     R_1 [0:4] --> <Rot3d [0.563679, 0.0939464, 0, 0.820634]>
 *     f_1 [5:6] --> 4.2
 *     f_2 [6:7] --> 4.3
 *     d_1 [7:8] --> 4.3
 *   >
 */
template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const Values<Scalar>& v);

}  // namespace sym

// Template method implementations
#include "./values.tcc"
