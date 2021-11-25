#ifndef _LCM_LIB_INLINE_H
#define _LCM_LIB_INLINE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// There are two optimizations in this file that aren't present in the original lcm_coretypes.h.
// They are on by default, and can be disabled with the following macros

// #define SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT
// Rewrite the endianness conversions to help GCCs emit better code (a single bswap or rev
// instruction)

// #define SKYDIO_DISABLE_LCM_FORCE_INLINE
// Add the force inline attribute on the encode/decode array methods. Without this they typically
// aren't inlined, but getting them inlined was a substantial win on my benchmarks.

#define SKYDIO_DISABLE_LCM_FORCE_INLINE 1
#if defined(SKYDIO_DISABLE_LCM_FORCE_INLINE)
// #define SKYDIO_LCM_INLINE_PREFIX __attribute__((unused))
#define SKYDIO_LCM_INLINE_PREFIX
#else
#define SKYDIO_LCM_INLINE_PREFIX __attribute__((unused, always_inline))
#endif

#ifdef __cplusplus

// Suppress warnings about C-style casts, since this code needs to build in
// both C and C++ modes
#if defined(__GNUC__) && defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

extern "C" {
#endif

union float_uint32 {
    float f;
    uint32_t i;
};

union double_uint64 {
    double f;
    uint64_t i;
};

typedef struct ___lcm_hash_ptr __lcm_hash_ptr;
struct ___lcm_hash_ptr {
    const __lcm_hash_ptr *parent;
    int64_t (*v)(void);
};

/**
 * BOOLEAN
 */
#define __boolean_hash_recursive __int8_t_hash_recursive
#define __boolean_decode_array_cleanup __int8_t_decode_array_cleanup
#define __boolean_encoded_array_size __int8_t_encoded_array_size
#define __boolean_encode_array __int8_t_encode_array
#define __boolean_decode_array __int8_t_decode_array
#define __boolean_clone_array __int8_t_clone_array
#define boolean_encoded_size int8_t_encoded_size

/**
 * BYTE
 */
#define __byte_hash_recursive __uint8_t_hash_recursive
#define __byte_decode_array_cleanup __uint8_t_decode_array_cleanup
#define __byte_encoded_array_size __uint8_t_encoded_array_size
#define __byte_encode_array __uint8_t_encode_array
#define __byte_decode_array __uint8_t_decode_array
#define __byte_clone_array __uint8_t_clone_array
#define byte_encoded_size uint8_t_encoded_size

/**
 * INT8_T
 */
#define __int8_t_hash_recursive(p) 0
#define __int8_t_decode_array_cleanup(p, sz) \
    {                                        \
    }
#define int8_t_encoded_size(p) (sizeof(int64_t) + sizeof(int8_t))

static inline int __int8_t_encoded_array_size(const int8_t *p, int elements)
{
    (void) p;
    return sizeof(int8_t) * elements;
}

static inline int __int8_t_encode_array(void *_buf, int offset, int maxlen, const int8_t *p,
                                        int elements)
{
    if (maxlen < elements)
        return -1;

    int8_t *buf = (int8_t *) _buf;
    memcpy(&buf[offset], p, elements);

    return elements;
}

static inline int __int8_t_decode_array(const void *_buf, int offset, int maxlen, int8_t *p,
                                        int elements)
{
    if (maxlen < elements)
        return -1;

    const int8_t *buf = (const int8_t *) _buf;
    memcpy(p, &buf[offset], elements);

    return elements;
}

static inline int __int8_t_clone_array(const int8_t *p, int8_t *q, int elements)
{
    memcpy(q, p, elements * sizeof(int8_t));
    return 0;
}

/**
 * UINT8_T
 */
#define __uint8_t_hash_recursive(p) 0
#define __uint8_t_decode_array_cleanup(p, sz) \
    {                                         \
    }
#define uint8_t_encoded_size(p) (sizeof(int64_t) + sizeof(uint8_t))

static inline int __uint8_t_encoded_array_size(const uint8_t *p, int elements)
{
    (void) p;
    return sizeof(uint8_t) * elements;
}

static inline int __uint8_t_encode_array(void *_buf, int offset, int maxlen, const uint8_t *p,
                                         int elements)
{
    if (maxlen < elements)
        return -1;

    uint8_t *buf = (uint8_t *) _buf;
    memcpy(&buf[offset], p, elements);

    return elements;
}

static inline int __uint8_t_decode_array(const void *_buf, int offset, int maxlen, uint8_t *p,
                                         int elements)
{
    if (maxlen < elements)
        return -1;

    const uint8_t *buf = (const uint8_t *) _buf;
    memcpy(p, &buf[offset], elements);

    return elements;
}

static inline int __uint8_t_clone_array(const uint8_t *p, uint8_t *q, int elements)
{
    memcpy(q, p, elements * sizeof(uint8_t));
    return 0;
}

/**
 * INT16_T
 */
#define __int16_t_hash_recursive(p) 0
#define __int16_t_decode_array_cleanup(p, sz) \
    {                                         \
    }
#define int16_t_encoded_size(p) (sizeof(int64_t) + sizeof(int16_t))

static inline int __int16_t_encoded_array_size(const int16_t *p, int elements)
{
    (void) p;
    return sizeof(int16_t) * elements;
}

static SKYDIO_LCM_INLINE_PREFIX int __int16_t_encode_array(void *_buf, int offset, int maxlen, const int16_t *p,
                                         int elements)
{
    int total_size = sizeof(int16_t) * elements;
    uint8_t *buf = (uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  use uint for shifting instead if int
    const uint16_t *unsigned_p = (const uint16_t *) p;
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        uint16_t v = unsigned_p[element];
        buf[pos++] = (v >> 8) & 0xff;
        buf[pos++] = (v & 0xff);
#else
        uint16_t value = unsigned_p[element];
        value = ((value & 0xff00) >> 8) | ((value & 0xff) << 8);
        memcpy(buf + pos, &value, sizeof(value));
        pos += 2;
#endif
    }

    return total_size;
}

static SKYDIO_LCM_INLINE_PREFIX int __int16_t_decode_array(const void *_buf, int offset, int maxlen, int16_t *p,
                                         int elements)
{
    int total_size = sizeof(int16_t) * elements;
    const uint8_t *buf = (const uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        p[element] = (buf[pos] << 8) + buf[pos + 1];
        pos += 2;
#else
        uint16_t value;
        memcpy(&value, buf + pos, sizeof(value));
        value = ((value & 0xff00) >> 8) | ((value & 0xff) << 8);
        p[element] = value;
        pos += 2;
#endif
    }

    return total_size;
}

static inline int __int16_t_clone_array(const int16_t *p, int16_t *q, int elements)
{
    memcpy(q, p, elements * sizeof(int16_t));
    return 0;
}

/**
 * UINT16_T
 */
#define __uint16_t_hash_recursive(p) 0
#define __uint16_t_decode_array_cleanup(p, sz) \
    {                                          \
    }
#define uint16_t_encoded_size(p) (sizeof(int64_t) + sizeof(uint16_t))

static inline int __uint16_t_encoded_array_size(const uint16_t *p, int elements)
{
    (void) p;
    return sizeof(uint16_t) * elements;
}

static SKYDIO_LCM_INLINE_PREFIX int __uint16_t_encode_array(void *_buf, int offset, int maxlen, const uint16_t *p,
                                          int elements)
{
    int total_size = sizeof(uint16_t) * elements;
    uint8_t *buf = (uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  use uint for shifting instead if int
    const uint16_t *unsigned_p = (uint16_t *) p;
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        uint16_t v = unsigned_p[element];
        buf[pos++] = (v >> 8) & 0xff;
        buf[pos++] = (v & 0xff);
#else
        uint16_t value = unsigned_p[element];
        value = ((value & 0xff00) >> 8) | ((value & 0xff) << 8);
        memcpy(buf + pos, &value, sizeof(value));
        pos += 2;
#endif
    }

    return total_size;
}

static SKYDIO_LCM_INLINE_PREFIX int __uint16_t_decode_array(const void *_buf, int offset, int maxlen, uint16_t *p,
                                          int elements)
{
    int total_size = sizeof(uint16_t) * elements;
    const uint8_t *buf = (const uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        p[element] = (buf[pos] << 8) + buf[pos + 1];
        pos += 2;
#else
        uint16_t value;
        memcpy(&value, buf + pos, sizeof(value));
        value = ((value & 0xff00) >> 8) | ((value & 0xff) << 8);
        p[element] = value;
        pos += 2;
#endif
    }

    return total_size;
}

static inline int __uint16_t_clone_array(const uint16_t *p, uint16_t *q, int elements)
{
    memcpy(q, p, elements * sizeof(uint16_t));
    return 0;
}

/**
 * INT32_T
 */
#define __int32_t_hash_recursive(p) 0
#define __int32_t_decode_array_cleanup(p, sz) \
    {                                         \
    }
#define int32_t_encoded_size(p) (sizeof(int64_t) + sizeof(int32_t))

static inline int __int32_t_encoded_array_size(const int32_t *p, int elements)
{
    (void) p;
    return sizeof(int32_t) * elements;
}

static SKYDIO_LCM_INLINE_PREFIX int __int32_t_encode_array(void *_buf, int offset, int maxlen, const int32_t *p,
                                         int elements)
{
    int total_size = sizeof(int32_t) * elements;
    uint8_t *buf = (uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  use uint for shifting instead if int
    const uint32_t *unsigned_p = (const uint32_t *) p;
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        const uint32_t v = unsigned_p[element];
        buf[pos++] = (v >> 24) & 0xff;
        buf[pos++] = (v >> 16) & 0xff;
        buf[pos++] = (v >> 8) & 0xff;
        buf[pos++] = (v & 0xff);
#else
        uint32_t value = unsigned_p[element];
        value =
            ((value & 0xFF000000u) >> 24u) |
            ((value & 0x00FF0000u) >> 8u) |
            ((value & 0x0000FF00u) << 8u) |
            ((value & 0x000000FFu) << 24u);
        memcpy(buf + pos, &value, sizeof(value));
        pos += 4;
#endif
    }

    return total_size;
}

static SKYDIO_LCM_INLINE_PREFIX int __int32_t_decode_array(const void *_buf, int offset, int maxlen, int32_t *p,
                                         int elements)
{
    int total_size = sizeof(int32_t) * elements;
    const uint8_t *buf = (const uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  use uint for shifting instead if int
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        p[element] = (((uint32_t) buf[pos + 0]) << 24) + (((uint32_t) buf[pos + 1]) << 16) +
                     (((uint32_t) buf[pos + 2]) << 8) + ((uint32_t) buf[pos + 3]);
        pos += 4;
#else
        uint32_t value;
        memcpy(&value, buf + pos, sizeof(value));
        value =
            ((value & 0xFF000000u) >> 24u) |
            ((value & 0x00FF0000u) >> 8u) |
            ((value & 0x0000FF00u) << 8u) |
            ((value & 0x000000FFu) << 24u);
            p[element] = value;
        pos += 4;
#endif
    }

    return total_size;
}

static inline int __int32_t_clone_array(const int32_t *p, int32_t *q, int elements)
{
    memcpy(q, p, elements * sizeof(int32_t));
    return 0;
}

/**
 * UINT32_T
 */
#define __uint32_t_hash_recursive(p) 0
#define __uint32_t_decode_array_cleanup(p, sz) \
    {                                          \
    }
#define uint32_t_encoded_size(p) (sizeof(int64_t) + sizeof(uint32_t))

static inline int __uint32_t_encoded_array_size(const uint32_t *p, int elements)
{
    (void) p;
    return sizeof(uint32_t) * elements;
}

static SKYDIO_LCM_INLINE_PREFIX int __uint32_t_encode_array(void *_buf, int offset, int maxlen, const uint32_t *p,
                                          int elements)
{
    int total_size = sizeof(uint32_t) * elements;
    uint8_t *buf = (uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  use uint for shifting instead if int
    const uint32_t *unsigned_p = (uint32_t *) p;
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        const uint32_t v = unsigned_p[element];
        buf[pos++] = (v >> 24) & 0xff;
        buf[pos++] = (v >> 16) & 0xff;
        buf[pos++] = (v >> 8) & 0xff;
        buf[pos++] = (v & 0xff);
#else
        uint32_t value = unsigned_p[element];
        value =
            ((value & 0xFF000000u) >> 24u) |
            ((value & 0x00FF0000u) >> 8u) |
            ((value & 0x0000FF00u) << 8u) |
            ((value & 0x000000FFu) << 24u);
        memcpy(buf + pos, &value, sizeof(value));
        pos += 4;
#endif
    }

    return total_size;
}

static SKYDIO_LCM_INLINE_PREFIX int __uint32_t_decode_array(const void *_buf, int offset, int maxlen, uint32_t *p,
                                          int elements)
{
    int total_size = sizeof(uint32_t) * elements;
    const uint8_t *buf = (const uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  use uint for shifting instead if int
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        p[element] = (((uint32_t) buf[pos + 0]) << 24) + (((uint32_t) buf[pos + 1]) << 16) +
                     (((uint32_t) buf[pos + 2]) << 8) + ((uint32_t) buf[pos + 3]);
        pos += 4;
#else
        uint32_t value;
        memcpy(&value, buf + pos, sizeof(value));
        value =
            ((value & 0xFF000000u) >> 24u) |
            ((value & 0x00FF0000u) >> 8u) |
            ((value & 0x0000FF00u) << 8u) |
            ((value & 0x000000FFu) << 24u);
        p[element] = value;
        pos += 4;
#endif
    }

    return total_size;
}

static inline int __uint32_t_clone_array(const uint32_t *p, uint32_t *q, int elements)
{
    memcpy(q, p, elements * sizeof(uint32_t));
    return 0;
}

/**
 * INT64_T
 */
#define __int64_t_hash_recursive(p) 0
#define __int64_t_decode_array_cleanup(p, sz) \
    {                                         \
    }
#define int64_t_encoded_size(p) (sizeof(int64_t) + sizeof(int64_t))

static inline int __int64_t_encoded_array_size(const int64_t *p, int elements)
{
    (void) p;
    return sizeof(int64_t) * elements;
}

static SKYDIO_LCM_INLINE_PREFIX int __int64_t_encode_array(void *_buf, int offset, int maxlen, const int64_t *p,
                                         int elements)
{
    int total_size = sizeof(int64_t) * elements;
    uint8_t *buf = (uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  We have now switched to using byte array access instead of shifting to reorder the bytes.
    // TODO(justin): This will only work properly on little-endian machines; we should add a
    // little-endian assert.
    // TODO(abe): WTF are we doing this instead of shifting like we do for the rest of the types?
    // Justin's original commit messages and gerrit review don't shed any light, but think it
    // *might* have been performance related?
    const uint64_t *unsigned_p = (uint64_t *) p;
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        const uint8_t *bytes = (const uint8_t *) (p + element);
        buf[pos++] = bytes[7];
        buf[pos++] = bytes[6];
        buf[pos++] = bytes[5];
        buf[pos++] = bytes[4];
        buf[pos++] = bytes[3];
        buf[pos++] = bytes[2];
        buf[pos++] = bytes[1];
        buf[pos++] = bytes[0];
#else
        uint64_t value = unsigned_p[element];
        value =
            ((value & 0xFF00000000000000u) >> 56u) |
            ((value & 0x00FF000000000000u) >> 40u) |
            ((value & 0x0000FF0000000000u) >> 24u) |
            ((value & 0x000000FF00000000u) >>  8u) |
            ((value & 0x00000000FF000000u) <<  8u) |
            ((value & 0x0000000000FF0000u) << 24u) |
            ((value & 0x000000000000FF00u) << 40u) |
            ((value & 0x00000000000000FFu) << 56u);
        memcpy(buf + pos, &value, sizeof(value));
        pos += 8;
#endif
    }

    return total_size;
}

static SKYDIO_LCM_INLINE_PREFIX int __int64_t_decode_array(const void *_buf, int offset, int maxlen, int64_t *p,
                                         int elements)
{
    int total_size = sizeof(int64_t) * elements;
    const uint8_t *buf = (const uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  use uint for shifting instead if int
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        uint64_t a = (((uint32_t) buf[pos + 0]) << 24) + (((uint32_t) buf[pos + 1]) << 16) +
                     (((uint32_t) buf[pos + 2]) << 8) + (uint32_t) buf[pos + 3];
        pos += 4;
        uint64_t b = (((uint32_t) buf[pos + 0]) << 24) + (((uint32_t) buf[pos + 1]) << 16) +
                     (((uint32_t) buf[pos + 2]) << 8) + (uint32_t) buf[pos + 3];
        pos += 4;
        p[element] = (a << 32) + (b & 0xffffffff);
#else
        uint64_t value;
        memcpy(&value, buf + pos, sizeof(value));
        value =
            ((value & 0xFF00000000000000u) >> 56u) |
            ((value & 0x00FF000000000000u) >> 40u) |
            ((value & 0x0000FF0000000000u) >> 24u) |
            ((value & 0x000000FF00000000u) >>  8u) |
            ((value & 0x00000000FF000000u) <<  8u) |
            ((value & 0x0000000000FF0000u) << 24u) |
            ((value & 0x000000000000FF00u) << 40u) |
            ((value & 0x00000000000000FFu) << 56u);
        p[element] = value;
        pos += 8;
#endif
    }

    return total_size;
}

static inline int __int64_t_clone_array(const int64_t *p, int64_t *q, int elements)
{
    memcpy(q, p, elements * sizeof(int64_t));
    return 0;
}

/**
 * UINT64_T
 */
#define __uint64_t_hash_recursive(p) 0
#define __uint64_t_decode_array_cleanup(p, sz) \
    {                                          \
    }
#define uint64_t_encoded_size(p) (sizeof(int64_t) + sizeof(uint64_t))

static inline int __uint64_t_encoded_array_size(const uint64_t *p, int elements)
{
    (void) p;
    return sizeof(uint64_t) * elements;
}

static SKYDIO_LCM_INLINE_PREFIX int __uint64_t_encode_array(void *_buf, int offset, int maxlen, const uint64_t *p,
                                          int elements)
{
    int total_size = sizeof(uint64_t) * elements;
    uint8_t *buf = (uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  We have now switched to using byte array access instead of shifting to reorder the bytes.
    // TODO(justin): This will only work properly on little-endian machines; we should add a
    // little-endian assert.
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        const uint8_t *bytes = (uint8_t *) (p + element);
        buf[pos++] = bytes[7];
        buf[pos++] = bytes[6];
        buf[pos++] = bytes[5];
        buf[pos++] = bytes[4];
        buf[pos++] = bytes[3];
        buf[pos++] = bytes[2];
        buf[pos++] = bytes[1];
        buf[pos++] = bytes[0];
#else
        uint64_t value = p[element];
        value =
            ((value & 0xFF00000000000000u) >> 56u) |
            ((value & 0x00FF000000000000u) >> 40u) |
            ((value & 0x0000FF0000000000u) >> 24u) |
            ((value & 0x000000FF00000000u) >>  8u) |
            ((value & 0x00000000FF000000u) <<  8u) |
            ((value & 0x0000000000FF0000u) << 24u) |
            ((value & 0x000000000000FF00u) << 40u) |
            ((value & 0x00000000000000FFu) << 56u);
        memcpy(buf + pos, &value, sizeof(value));
        pos += 8;
#endif
    }

    return total_size;
}

static SKYDIO_LCM_INLINE_PREFIX int __uint64_t_decode_array(const void *_buf, int offset, int maxlen, uint64_t *p,
                                          int elements)
{
    int total_size = sizeof(uint64_t) * elements;
    uint8_t *buf = (uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  use uint for shifting instead if int
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        uint64_t a = (((uint32_t) buf[pos + 0]) << 24) + (((uint32_t) buf[pos + 1]) << 16) +
                     (((uint32_t) buf[pos + 2]) << 8) + (uint32_t) buf[pos + 3];
        pos += 4;
        uint64_t b = (((uint32_t) buf[pos + 0]) << 24) + (((uint32_t) buf[pos + 1]) << 16) +
                     (((uint32_t) buf[pos + 2]) << 8) + (uint32_t) buf[pos + 3];
        pos += 4;
        p[element] = (a << 32) + (b & 0xffffffff);
#else
        uint64_t value;
        memcpy(&value, buf + pos, sizeof(value));
        value =
            ((value & 0xFF00000000000000u) >> 56u) |
            ((value & 0x00FF000000000000u) >> 40u) |
            ((value & 0x0000FF0000000000u) >> 24u) |
            ((value & 0x000000FF00000000u) >>  8u) |
            ((value & 0x00000000FF000000u) <<  8u) |
            ((value & 0x0000000000FF0000u) << 24u) |
            ((value & 0x000000000000FF00u) << 40u) |
            ((value & 0x00000000000000FFu) << 56u);
        p[element] = value;
        pos += 8;
#endif
    }

    return total_size;
}

static inline int __uint64_t_clone_array(const uint64_t *p, uint64_t *q, int elements)
{
    memcpy(q, p, elements * sizeof(uint64_t));
    return 0;
}

/**
 * FLOAT
 */
#define __float_hash_recursive(p) 0
#define __float_decode_array_cleanup(p, sz) \
    {                                       \
    }
#define float_encoded_size(p) (sizeof(int64_t) + sizeof(float))

static inline int __float_encoded_array_size(const float *p, int elements)
{
    (void) p;
    return sizeof(float) * elements;
}

static SKYDIO_LCM_INLINE_PREFIX int __float_encode_array(void *_buf, int offset, int maxlen, const float *p,
                                       int elements)
{
    int total_size = sizeof(float) * elements;
    uint8_t *buf = (uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  use uint for shifting instead if int
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        uint32_t v;
        memcpy(&v, &p[element], sizeof(v));
        buf[pos++] = (v >> 24) & 0xff;
        buf[pos++] = (v >> 16) & 0xff;
        buf[pos++] = (v >> 8) & 0xff;
        buf[pos++] = (v & 0xff);
#else
        uint32_t value;
        memcpy(&value, &p[element], sizeof(value));
        value =
            ((value & 0xFF000000u) >> 24u) |
            ((value & 0x00FF0000u) >> 8u) |
            ((value & 0x0000FF00u) << 8u) |
            ((value & 0x000000FFu) << 24u);
        memcpy(buf + pos, &value, sizeof(value));
        pos += 4;
#endif
    }

    return total_size;
}

static SKYDIO_LCM_INLINE_PREFIX int __float_decode_array(const void *_buf, int offset, int maxlen, float *p,
                                       int elements)
{
    int total_size = sizeof(float) * elements;
    const uint8_t *buf = (const uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  use uint for shifting instead if int
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        uint32_t v = (((uint32_t) buf[pos + 0]) << 24) + (((uint32_t) buf[pos + 1]) << 16) +
                     (((uint32_t) buf[pos + 2]) << 8) + ((uint32_t) buf[pos + 3]);
        memcpy(&p[element], &v, sizeof(v));
        pos += 4;
#else
        uint32_t value;
        memcpy(&value, buf + pos, sizeof(value));
        value =
            ((value & 0xFF000000u) >> 24u) |
            ((value & 0x00FF0000u) >> 8u) |
            ((value & 0x0000FF00u) << 8u) |
            ((value & 0x000000FFu) << 24u);
        memcpy(&p[element], &value, sizeof(value));
        pos += 4;
#endif
    }

    return total_size;
}

static inline int __float_clone_array(const float *p, float *q, int elements)
{
    memcpy(q, p, elements * sizeof(float));
    return 0;
}

/**
 * DOUBLE
 */
#define __double_hash_recursive(p) 0
#define __double_decode_array_cleanup(p, sz) \
    {                                        \
    }
#define double_encoded_size(p) (sizeof(int64_t) + sizeof(double))

static inline int __double_encoded_array_size(const double *p, int elements)
{
    (void) p;
    return sizeof(double) * elements;
}

static SKYDIO_LCM_INLINE_PREFIX int __double_encode_array(void *_buf, int offset, int maxlen, const double *p,
                                        int elements)
{
    int total_size = sizeof(double) * elements;
    uint8_t *buf = (uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  We have now switched to using byte array access instead of shifting to reorder the bytes.
    // TODO(justin): This will only work properly on little-endian machines; we should add a
    // little-endian assert.
    // TODO(abe): WTF are we doing this instead of shifting like we do for the rest of the types?
    // Justin's original commit messages and gerrit review don't shed any light, but think it
    // *might* have been performance related?
    const uint64_t *unsigned_p = (uint64_t *) p;
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        const uint8_t *bytes = (const uint8_t *) (p + element);
        buf[pos++] = bytes[7];
        buf[pos++] = bytes[6];
        buf[pos++] = bytes[5];
        buf[pos++] = bytes[4];
        buf[pos++] = bytes[3];
        buf[pos++] = bytes[2];
        buf[pos++] = bytes[1];
        buf[pos++] = bytes[0];
#else
        uint64_t value;
        memcpy(&value, &p[element], sizeof(value));
        value =
            ((value & 0xFF00000000000000u) >> 56u) |
            ((value & 0x00FF000000000000u) >> 40u) |
            ((value & 0x0000FF0000000000u) >> 24u) |
            ((value & 0x000000FF00000000u) >>  8u) |
            ((value & 0x00000000FF000000u) <<  8u) |
            ((value & 0x0000000000FF0000u) << 24u) |
            ((value & 0x000000000000FF00u) << 40u) |
            ((value & 0x00000000000000FFu) << 56u);
        memcpy(buf + pos, &value, sizeof(value));
        pos += 8;
#endif
    }

    return total_size;
}

static SKYDIO_LCM_INLINE_PREFIX int __double_decode_array(const void *_buf, int offset, int maxlen, double *p,
                                        int elements)
{
    int total_size = sizeof(double) * elements;
    const uint8_t *buf = (const uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    //  See Section 5.8 paragraph 3 of the standard
    //  http://open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4527.pdf
    //  use uint for shifting instead if int
    for (element = 0; element < elements; element++) {
#if defined(SKYDIO_DISABLE_LCM_IMPROVED_ENDIAN_HINT)
        uint64_t a = (((uint32_t) buf[pos + 0]) << 24) + (((uint32_t) buf[pos + 1]) << 16) +
                     (((uint32_t) buf[pos + 2]) << 8) + (uint32_t) buf[pos + 3];
        pos += 4;
        uint64_t b = (((uint32_t) buf[pos + 0]) << 24) + (((uint32_t) buf[pos + 1]) << 16) +
                     (((uint32_t) buf[pos + 2]) << 8) + (uint32_t) buf[pos + 3];
        pos += 4;
        uint64_t v = (a << 32) + (b & 0xffffffff);
        memcpy(&p[element], &v, sizeof(v));
#else
        uint64_t value;
        memcpy(&value, buf + pos, sizeof(value));
        value =
            ((value & 0xFF00000000000000u) >> 56u) |
            ((value & 0x00FF000000000000u) >> 40u) |
            ((value & 0x0000FF0000000000u) >> 24u) |
            ((value & 0x000000FF00000000u) >>  8u) |
            ((value & 0x00000000FF000000u) <<  8u) |
            ((value & 0x0000000000FF0000u) << 24u) |
            ((value & 0x000000000000FF00u) << 40u) |
            ((value & 0x00000000000000FFu) << 56u);
        memcpy(&p[element], &value, sizeof(value));
        pos += 8;
#endif
    }

    return total_size;
}

static inline int __double_clone_array(const double *p, double *q, int elements)
{
    memcpy(q, p, elements * sizeof(double));
    return 0;
}

/**
 * STRING
 */
#define __string_hash_recursive(p) 0

static inline int __string_decode_array_cleanup(char **s, int elements)
{
    int element;
    for (element = 0; element < elements; element++)
        free(s[element]);
    return 0;
}

static inline int __string_encoded_array_size(char *const *s, int elements)
{
    int size = 0;
    int element;
    for (element = 0; element < elements; element++)
        size += 4 + strlen(s[element]) + 1;

    return size;
}

static inline int __string_encoded_size(char *const *s)
{
    return sizeof(int64_t) + __string_encoded_array_size(s, 1);
}

static inline int __string_encode_array(void *_buf, int offset, int maxlen, char *const *p,
                                        int elements)
{
    int pos = 0, thislen;
    int element;

    for (element = 0; element < elements; element++) {
        int32_t length = strlen(p[element]) + 1;  // length includes \0

        thislen = __int32_t_encode_array(_buf, offset + pos, maxlen - pos, &length, 1);
        if (thislen < 0)
            return thislen;
        else
            pos += thislen;

        thislen =
            __int8_t_encode_array(_buf, offset + pos, maxlen - pos, (int8_t *) p[element], length);
        if (thislen < 0)
            return thislen;
        else
            pos += thislen;
    }

    return pos;
}

static inline int __string_decode_array(const void *_buf, int offset, int maxlen, char **p,
                                        int elements)
{
    int pos = 0, thislen;
    int element;

    for (element = 0; element < elements; element++) {
        int32_t length;

        // read length including \0
        thislen = __int32_t_decode_array(_buf, offset + pos, maxlen - pos, &length, 1);
        if (thislen < 0)
            return thislen;
        else
            pos += thislen;

        p[element] = (char *) malloc(length);
        thislen =
            __int8_t_decode_array(_buf, offset + pos, maxlen - pos, (int8_t *) p[element], length);
        if (thislen < 0)
            return thislen;
        else
            pos += thislen;
    }

    return pos;
}

static inline int __string_clone_array(char *const *p, char **q, int elements)
{
    int element;
    for (element = 0; element < elements; element++) {
        // because strdup is not C99
        size_t len = strlen(p[element]) + 1;
        q[element] = (char *) malloc(len);
        memcpy(q[element], p[element], len);
    }
    return 0;
}

static inline void *lcm_malloc(size_t sz)
{
    if (sz)
        return malloc(sz);
    return NULL;
}

/**
 * Describes the type of a single field in an LCM message.
 */
typedef enum {
    LCM_FIELD_INT8_T,
    LCM_FIELD_INT16_T,
    LCM_FIELD_INT32_T,
    LCM_FIELD_INT64_T,
    LCM_FIELD_BYTE,
    LCM_FIELD_FLOAT,
    LCM_FIELD_DOUBLE,
    LCM_FIELD_STRING,
    LCM_FIELD_BOOLEAN,
    LCM_FIELD_USER_TYPE
} lcm_field_type_t;

#define LCM_TYPE_FIELD_MAX_DIM 50

/**
 * Describes a single lcmtype field's datatype and array dimmensions
 */
typedef struct _lcm_field_t lcm_field_t;
struct _lcm_field_t {
    /**
     * name of the field
     */
    const char *name;

    /**
     * datatype of the field
     **/
    lcm_field_type_t type;

    /**
     * datatype of the field (in string format)
     * this should be the same as in the lcm type decription file
     */
    const char *typestr;

    /**
     * number of array dimensions
     * if the field is scalar, num_dim should equal 0
     */
    int num_dim;

    /**
     * the size of each dimension. Valid on [0:num_dim-1].
     */
    int32_t dim_size[LCM_TYPE_FIELD_MAX_DIM];

    /**
     * a boolean describing whether the dimension is
     * variable. Valid on [0:num_dim-1].
     */
    int8_t dim_is_variable[LCM_TYPE_FIELD_MAX_DIM];

    /**
     * a data pointer to the start of this field
     */
    void *data;
};

typedef int (*lcm_encode_t)(void *buf, int offset, int maxlen, const void *p);
typedef int (*lcm_decode_t)(const void *buf, int offset, int maxlen, void *p);
typedef int (*lcm_decode_cleanup_t)(void *p);
typedef int (*lcm_encoded_size_t)(const void *p);
typedef int (*lcm_struct_size_t)(void);
typedef int (*lcm_num_fields_t)(void);
typedef int (*lcm_get_field_t)(const void *p, int i, lcm_field_t *f);
typedef int64_t (*lcm_get_hash_t)(void);

/**
 * Describes an lcmtype info, enabling introspection
 */
typedef struct _lcm_type_info_t lcm_type_info_t;
struct _lcm_type_info_t {
    lcm_encode_t encode;
    lcm_decode_t decode;
    lcm_decode_cleanup_t decode_cleanup;
    lcm_encoded_size_t encoded_size;
    lcm_struct_size_t struct_size;
    lcm_num_fields_t num_fields;
    lcm_get_field_t get_field;
    lcm_get_hash_t get_hash;
};

#ifdef __cplusplus
}
#if defined(__GNUC__) && defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif

#undef SKYDIO_LCM_INLINE_PREFIX

#endif
