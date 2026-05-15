#include <iterator>
#include <sstream>
#include <algorithm>
#include <symengine/printers/stringbox.h>

// Macro to let string literals be unicode const char in all C++ standards
// Otherwise u8"" would be char8_t in C++20
#define U8(x) reinterpret_cast<const char *>(u8##x)

namespace SymEngine
{

void StringBox::pad_lines(std::size_t new_width)
{
    auto diff = new_width - width_;
    auto half = diff / 2;
    auto odd = diff % 2;
    for (std::string &line : lines_) {
        line.insert(0, std::string(half + odd, ' '));
        if (half > 0) {
            line.append(std::string(half, ' '));
        }
    }
}

void StringBox::add_below(StringBox &other)
{
    if (other.width_ > width_) {
        pad_lines(other.width_);
        width_ = other.width_;
    } else if (width_ > other.width_) {
        other.pad_lines(width_);
        other.width_ = width_;
    }
    lines_.insert(lines_.end(), other.lines_.begin(), other.lines_.end());
}

void StringBox::add_below_unicode_line(StringBox &other)
{
    auto new_width = std::max(width_, other.width_);
    std::string bar;
    for (unsigned i = 0; i < new_width; i++) {
        bar.append(U8("\u2015"));
    }
    StringBox barbox(bar, new_width);
    add_below(barbox);
    add_below(other);
}

void StringBox::add_right(StringBox &other)
{
    StringBox *smaller;
    auto this_size = lines_.size();
    auto other_size = other.lines_.size();
    if (other_size > this_size) {
        smaller = this;
    } else {
        smaller = &other;
    }
    auto diff
        = std::max(other_size, this_size) - std::min(other_size, this_size);
    auto half = diff / 2;
    auto odd = diff % 2;
    std::string pad(smaller->width_, ' ');
    for (unsigned i = 0; i < half; i++) {
        smaller->lines_.push_back(pad);
        smaller->lines_.insert(smaller->lines_.begin(), pad);
    }
    if (odd == 1) {
        smaller->lines_.insert(lines_.begin(), pad);
    }
    for (unsigned i = 0; i < lines_.size(); i++) {
        lines_[i].append(other.lines_[i]);
    }
    width_ += other.width_;
}

void StringBox::add_power(StringBox &other)
{
    for (std::string &line : lines_) {
        line.append(std::string(other.width_, ' '));
    }
    for (std::string &line : other.lines_) {
        lines_.insert(lines_.begin(), std::string(width_, ' ') + line);
    }
    width_ += other.width_;
}

void StringBox::enclose_abs()
{
    for (std::string &line : lines_) {
        line.insert(0, U8("\u2502"));
        line.append(U8("\u2502"));
    }
    width_ += 2;
}

void StringBox::enclose_parens()
{
    add_left_parens();
    add_right_parens();
}

void StringBox::enclose_sqbrackets()
{
    add_left_sqbracket();
    add_right_sqbracket();
}

void StringBox::enclose_curlies()
{
    add_left_curly();
    add_right_curly();
}

void StringBox::add_left_parens()
{
    if (lines_.size() == 1) {
        lines_[0].insert(0, "(");
    } else {
        lines_[0].insert(0, U8("\u239B"));
        lines_.back().insert(0, U8("\u239D"));
        for (unsigned i = 1; i < lines_.size() - 1; i++) {
            lines_[i].insert(0, U8("\u239C"));
        }
    }
    width_ += 1;
}

void StringBox::add_right_parens()
{
    if (lines_.size() == 1) {
        lines_[0].append(")");
    } else {
        lines_[0].append(U8("\u239E"));
        lines_.back().append(U8("\u23A0"));
        for (unsigned i = 1; i < lines_.size() - 1; i++) {
            lines_[i].append(U8("\u239F"));
        }
    }
    width_ += 1;
}

void StringBox::add_left_sqbracket()
{
    if (lines_.size() == 1) {
        lines_[0].insert(0, "[");
    } else {
        lines_[0].insert(0, U8("\u23A1"));
        lines_.back().insert(0, U8("\u23A3"));
        for (unsigned i = 1; i < lines_.size() - 1; i++) {
            lines_[i].insert(0, U8("\u23A2"));
        }
    }
    width_ += 1;
}

void StringBox::add_left_curly()
{
    if (lines_.size() == 1) {
        lines_[0].insert(0, "{");
    } else if (lines_.size() == 2) {
        lines_[0].insert(0, U8("\u23A7"));
        lines_[1].insert(0, U8("\u23A9"));
        lines_.insert(lines_.begin() + 1,
                      U8("\u23A8") + std::string(width_, ' '));
    } else {
        lines_[0].insert(0, U8("\u23A7"));
        lines_.back().insert(0, U8("\u23A9"));
        std::size_t mid = lines_.size() / 2;
        for (std::size_t i = 1; i < lines_.size() - 1; i++) {
            if (i == mid) {
                lines_[i].insert(0, U8("\u23A8"));
            } else {
                lines_[i].insert(0, U8("\u23AA"));
            }
        }
    }
    width_ += 1;
}

void StringBox::add_right_curly()
{
    if (lines_.size() == 1) {
        lines_[0].append("}");
    } else if (lines_.size() == 2) {
        lines_[0].append(U8("\u23AB"));
        lines_[1].append(U8("\u23AD"));
        lines_.insert(lines_.begin() + 1,
                      std::string(width_, ' ') + U8("\u23AC"));
    } else {
        lines_[0].append(U8("\u23AB"));
        lines_.back().append(U8("\u23AD"));
        std::size_t mid = lines_.size() / 2;
        for (std::size_t i = 1; i < lines_.size() - 1; i++) {
            if (i == mid) {
                lines_[i].append(U8("\u23AC"));
            } else {
                lines_[i].append(U8("\u23AA"));
            }
        }
    }
    width_ += 1;
}

void StringBox::add_right_sqbracket()
{
    if (lines_.size() == 1) {
        lines_[0].append("]");
    } else {
        lines_[0].append(U8("\u23A4"));
        lines_.back().append(U8("\u23A5"));
        for (unsigned i = 1; i < lines_.size() - 1; i++) {
            lines_[i].append(U8("\u23A6"));
        }
    }
    width_ += 1;
}

void StringBox::enclose_floor()
{
    lines_.back().insert(0, U8("\u230A"));
    lines_.back().append(U8("\u230B"));
    for (unsigned i = 0; i < lines_.size() - 1; i++) {
        lines_[i].insert(0, U8("\u2502"));
        lines_[i].append(U8("\u2502"));
    }
    width_ += 2;
}

void StringBox::enclose_ceiling()
{
    lines_[0].insert(0, U8("\u2308"));
    lines_[0].append(U8("\u2309"));
    for (unsigned i = 1; i < lines_.size(); i++) {
        lines_[i].insert(0, U8("\u2502"));
        lines_[i].append(U8("\u2502"));
    }
    width_ += 2;
}

void StringBox::enclose_sqrt()
{
    std::size_t len = lines_.size();
    std::size_t i = len;
    for (std::string &line : lines_) {
        if (i == 1) {
            line.insert(0, U8("\u2572\u2571") + std::string(len - i, ' '));
        } else {
            line.insert(0, std::string(i, ' ') + U8("\u2571")
                               + std::string(len - i, ' '));
        }
        i--;
    }
    lines_.insert(lines_.begin(),
                  std::string(len + 1, ' ') + std::string(width_, '_'));
    width_ += len + 1;
}

std::string StringBox::get_string() const
{
    std::ostringstream os;
    auto b = begin(lines_), e = end(lines_);

    if (b != e) {
        std::copy(b, prev(e), std::ostream_iterator<std::string>(os, "\n"));
        b = prev(e);
    }
    if (b != e) {
        os << *b;
    }

    return os.str();
}

}; // namespace SymEngine
