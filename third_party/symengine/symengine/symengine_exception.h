#ifndef SYMENGINE_EXCEPTION_H
#define SYMENGINE_EXCEPTION_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SYMENGINE_NO_EXCEPTION = 0,
    SYMENGINE_RUNTIME_ERROR = 1,
    SYMENGINE_DIV_BY_ZERO = 2,
    SYMENGINE_NOT_IMPLEMENTED = 3,
    SYMENGINE_DOMAIN_ERROR = 4,
    SYMENGINE_PARSE_ERROR = 5,
} symengine_exceptions_t;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <exception>
#include <string>

namespace SymEngine
{

class SymEngineException : public std::exception
{
    std::string m_msg;
    symengine_exceptions_t ec;

public:
    SymEngineException(const std::string &msg, symengine_exceptions_t error)
        : m_msg(msg), ec(error)
    {
    }
    SymEngineException(const std::string &msg)
        : SymEngineException(msg, SYMENGINE_RUNTIME_ERROR)
    {
    }
    const char *what() const throw()
    {
        return m_msg.c_str();
    }
    symengine_exceptions_t error_code()
    {
        return ec;
    }
};

class DivisionByZeroError : public SymEngineException
{
public:
    DivisionByZeroError(const std::string &msg)
        : SymEngineException(msg, SYMENGINE_DIV_BY_ZERO)
    {
    }
};

class NotImplementedError : public SymEngineException
{
public:
    NotImplementedError(const std::string &msg)
        : SymEngineException(msg, SYMENGINE_NOT_IMPLEMENTED)
    {
    }
};

class DomainError : public SymEngineException
{
public:
    DomainError(const std::string &msg)
        : SymEngineException(msg, SYMENGINE_DOMAIN_ERROR)
    {
    }
};

class ParseError : public SymEngineException
{
public:
    ParseError(const std::string &msg)
        : SymEngineException(msg, SYMENGINE_PARSE_ERROR)
    {
    }
};
}
#endif // __cplusplus
#endif // SYMENGINE_EXCEPTION_H
