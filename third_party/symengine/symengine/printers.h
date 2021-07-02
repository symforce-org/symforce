#ifndef SYMENGINE_PRINTER_H
#define SYMENGINE_PRINTER_H

#include <symengine/basic.h>

namespace SymEngine
{
std::string str(const Basic &x);
std::string julia_str(const Basic &x);
std::string ascii_art();

std::string mathml(const Basic &x);

std::string latex(const Basic &x);

std::string ccode(const Basic &x);
std::string c89code(const Basic &x);
std::string c99code(const Basic &x);
std::string jscode(const Basic &x);
}

#endif // SYMENGINE_PRINTER_H
