// This file was hand-written, modified from the Matrix34d type, with the hash
// generated by skymarshal.
#pragma once
#include <utility>

#include <lcm/lcm_coretypes.h>
#ifdef SKYMARSHAL_PRINTING_ENABLED
#include <lcm/lcm_reflection_eigen.hpp>
#endif
#include <Eigen/Core>

namespace eigen_lcm {

class Matrix34d : public Eigen::Matrix<double, 3, 4, Eigen::DontAlign> {
 public:
  // Default constructor. Unlike Eigen, eigen_lcm types initialize themselves to zero.
  // This is to prevent all sorts of uninitialized memory footguns.
  // Since the X sized variants are zero-sized by default, it is OK to default construct them.
  Matrix34d() : Eigen::Matrix<double, 3, 4, Eigen::DontAlign>() {
    *this = Matrix34d::Zero();
  }

  // Pass through constructor
  template <typename... Args>
  Matrix34d(Args&&... args) : Eigen::Matrix<double, 3, 4, Eigen::DontAlign>(args...) {}

  // This method allows you to assign Eigen expressions to eigen_lcm::Matrix34d
  template <typename OtherDerived>
  eigen_lcm::Matrix34d& operator=(const Eigen::MatrixBase<OtherDerived>& other) {
    Eigen::Matrix<double, 3, 4, Eigen::DontAlign>::operator=(other);
    return *this;
  }

  // Stream formatting operator
  friend std::ostream& operator<<(std::ostream& os, const eigen_lcm::Matrix34d& obj) {
    os << "Matrix34d(";
    for (size_t i = 0; i < obj.size(); ++i) {
      os << obj.data()[i];
      if (i + 1 != obj.size()) {
        os << ", ";
      }
    }
    os << ")";
    return os;
  }

 private:
  // Disable comma initialization magic for eigen_lcm types
  template <typename T>
  Eigen::Matrix<double, 3, 4, Eigen::DontAlign>& operator<<(T other);

 public:
  /**
   * Encode a message into binary form.
   *
   * @param buf The output buffer.
   * @param offset Encoding starts at thie byte offset into @p buf.
   * @param maxlen Maximum number of bytes to write.  This should generally be
   *  equal to getEncodedSize().
   * @return The number of bytes encoded, or <0 on error.
   */
  inline __lcm_buffer_size encode(void* buf, __lcm_buffer_size offset,
                                  __lcm_buffer_size maxlen) const;

  /**
   * Check how many bytes are required to encode this message.
   */
  inline __lcm_buffer_size getEncodedSize() const;

  /**
   * Decode a message from binary form into this instance.
   *
   * @param buf The buffer containing the encoded message.
   * @param offset The byte offset into @p buf where the encoded message starts.
   * @param maxlen The maximum number of bytes to read while decoding.
   * @return The number of bytes decoded, or <0 if an error occured.
   */
  inline __lcm_buffer_size decode(const void* buf, __lcm_buffer_size offset,
                                  __lcm_buffer_size maxlen);

  /**
   * Retrieve the 64-bit fingerprint identifying the structure of the message.
   * Note that the fingerprint is the same for all instances of the same
   * message type, and is a fingerprint on the message type definition, not on
   * the message contents.
   */
  constexpr static uint64_t getHash();

  using type_name_array_t = const char[10];

  inline static constexpr type_name_array_t* getTypeNameArrayPtr();

  /**
   * Returns "Matrix34d"
   */
  inline static constexpr const char* getTypeName();

  using package_name_array_t = const char[10];

  inline static constexpr package_name_array_t* getPackageNameArrayPtr();

  /**
   * Returns "eigen_lcm"
   */
  inline static constexpr const char* getPackageName();

  // LCM support functions. Users should not call these
  inline __lcm_buffer_size _encodeNoHash(void* buf, __lcm_buffer_size offset,
                                         __lcm_buffer_size maxlen) const;
  inline __lcm_buffer_size _getEncodedSizeNoHash() const;
  inline __lcm_buffer_size _decodeNoHash(const void* buf, __lcm_buffer_size offset,
                                         __lcm_buffer_size maxlen);
  constexpr static uint64_t _computeHash(const __lcm_hash_ptr*) {
    uint64_t hash = 0x2b2f4ef42124f02eLL;
    return (hash << 1) + ((hash >> 63) & 1);
  }
};

__lcm_buffer_size Matrix34d::encode(void* buf, __lcm_buffer_size offset,
                                    __lcm_buffer_size maxlen) const {
  __lcm_buffer_size pos = 0, tlen;
  uint64_t hash = getHash();

  tlen = __uint64_t_encode_array(buf, offset + pos, maxlen - pos, &hash, 1);
  if (tlen < 0)
    return tlen;
  else
    pos += tlen;

  tlen = this->_encodeNoHash(buf, offset + pos, maxlen - pos);
  if (tlen < 0)
    return tlen;
  else
    pos += tlen;

  return pos;
}

__lcm_buffer_size Matrix34d::decode(const void* buf, __lcm_buffer_size offset,
                                    __lcm_buffer_size maxlen) {
  __lcm_buffer_size pos = 0, thislen;

  uint64_t hash;
  thislen = __uint64_t_decode_array(buf, offset + pos, maxlen - pos, &hash, 1);
  if (thislen < 0)
    return thislen;
  else
    pos += thislen;
  if (hash != getHash())
    return -1;

  thislen = this->_decodeNoHash(buf, offset + pos, maxlen - pos);
  if (thislen < 0)
    return thislen;
  else
    pos += thislen;

  return pos;
}

__lcm_buffer_size Matrix34d::getEncodedSize() const {
  return 8 + _getEncodedSizeNoHash();
}

constexpr uint64_t Matrix34d::getHash() {
  return _computeHash(NULL);
}

constexpr Matrix34d::type_name_array_t* Matrix34d::getTypeNameArrayPtr() {
  return &"Matrix34d";
}

constexpr const char* Matrix34d::getTypeName() {
  return *Matrix34d::getTypeNameArrayPtr();
}

constexpr Matrix34d::package_name_array_t* Matrix34d::getPackageNameArrayPtr() {
  return &"eigen_lcm";
}

constexpr const char* Matrix34d::getPackageName() {
  return *Matrix34d::getPackageNameArrayPtr();
}

__lcm_buffer_size Matrix34d::_encodeNoHash(void* buf, __lcm_buffer_size offset,
                                           __lcm_buffer_size maxlen) const {
  __lcm_buffer_size pos = 0, tlen;

  tlen = __double_encode_array(buf, offset + pos, maxlen - pos, this->data(), 12);
  if (tlen < 0)
    return tlen;
  else
    pos += tlen;

  return pos;
}

__lcm_buffer_size Matrix34d::_decodeNoHash(const void* buf, __lcm_buffer_size offset,
                                           __lcm_buffer_size maxlen) {
  __lcm_buffer_size pos = 0, tlen;

  tlen = __double_decode_array(buf, offset + pos, maxlen - pos, this->data(), 12);
  if (tlen < 0)
    return tlen;
  else
    pos += tlen;

  return pos;
}

__lcm_buffer_size Matrix34d::_getEncodedSizeNoHash() const {
  __lcm_buffer_size enc_size = 0;
  enc_size += __double_encoded_array_size(NULL, 12);
  return enc_size;
}

}  // namespace eigen_lcm