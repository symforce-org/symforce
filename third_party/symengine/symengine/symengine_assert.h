#ifndef SYMENGINE_ASSERT_H
#define SYMENGINE_ASSERT_H

// SYMENGINE_ASSERT uses internal functions to perform as assert
// so that there is no effect with NDEBUG
#if defined(WITH_SYMENGINE_ASSERT)

#if !defined(SYMENGINE_ASSERT)
#define stringize(s) #s
#define XSTR(s) stringize(s)
#define SYMENGINE_ASSERT(cond)                                                 \
    {                                                                          \
        if (!(cond)) {                                                         \
            std::cerr << "SYMENGINE_ASSERT failed: " << __FILE__               \
                      << "\nfunction " << __func__ << "(), line number "       \
                      << __LINE__ << " at \n"                                  \
                      << XSTR(cond) << "\n";                                   \
            abort();                                                           \
        }                                                                      \
    }
#endif // !defined(SYMENGINE_ASSERT)

#if !defined(SYMENGINE_ASSERT_MSG)
#define SYMENGINE_ASSERT_MSG(cond, msg)                                        \
    {                                                                          \
        if (!(cond)) {                                                         \
            std::cerr << "SYMENGINE_ASSERT failed: " << __FILE__               \
                      << "\nfunction " << __func__ << "(), line number "       \
                      << __LINE__ << " at \n"                                  \
                      << XSTR(cond) << "\n"                                    \
                      << "ERROR MESSAGE:\n"                                    \
                      << msg << "\n";                                          \
            abort();                                                           \
        }                                                                      \
    }
#endif // !defined(SYMENGINE_ASSERT_MSG)

#else // defined(WITH_SYMENGINE_ASSERT)

#define SYMENGINE_ASSERT(cond)
#define SYMENGINE_ASSERT_MSG(cond, msg)

#endif // defined(WITH_SYMENGINE_ASSERT)

#define SYMENGINE_ERROR(description)                                           \
    std::cerr << description;                                                  \
    std::cerr << "\n";                                                         \
    abort();

#endif // SYMENGINE_ASSERT_H
