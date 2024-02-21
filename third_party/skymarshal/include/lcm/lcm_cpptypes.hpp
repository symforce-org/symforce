#pragma once

#include <lcm/lcm_coretypes.h>

// Double check assumptions about representation of bools on this platform.
static_assert(sizeof(bool) == 1, "bool is not 1 byte???");

/**
 * BOOLEAN
 */
#define __boolean_hash_recursive __int8_t_hash_recursive
#define __boolean_decode_array_cleanup __int8_t_decode_array_cleanup
#define boolean_encoded_size int8_t_encoded_size

static inline __lcm_buffer_size __boolean_encoded_array_size(const bool *p,
                                                             __lcm_buffer_size elements)
{
    (void) p;
    return sizeof(bool) * elements;
}

static inline __lcm_buffer_size __boolean_encode_array(void *_buf, __lcm_buffer_size offset,
                                                       __lcm_buffer_size maxlen, const bool *p,
                                                       __lcm_buffer_size elements)
{
    if (maxlen < elements)
        return -1;

    bool *buf = (bool *) _buf;
    memcpy(&buf[offset], p, elements);

    return elements;
}

static inline __lcm_buffer_size __boolean_decode_array(const void *_buf, __lcm_buffer_size offset,
                                                       __lcm_buffer_size maxlen, bool *p,
                                                       __lcm_buffer_size elements)
{
    if (maxlen < elements)
        return -1;

    const bool *buf = (const bool *) _buf;
    memcpy(p, &buf[offset], elements);

    return elements;
}

static inline __lcm_buffer_size __boolean_clone_array(const bool *p, bool *q,
                                                      __lcm_buffer_size elements)
{
    memcpy(q, p, elements * sizeof(bool));
    return 0;
}
