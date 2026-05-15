#include <string>
#include <vector>

namespace SymEngine
{

class StringBox
{
private:
    std::vector<std::string> lines_;
    std::size_t width_;

    void pad_lines(std::size_t new_width);

public:
    explicit StringBox(std::string s)
    {
        lines_.push_back(s);
        width_ = s.length();
    }

    StringBox(std::string s, std::size_t width)
    {
        lines_.push_back(s);
        width_ = width;
    }

    StringBox()
    {
        width_ = 0;
    }

    std::string get_string() const;
    void add_below(StringBox &other);
    void add_below_unicode_line(StringBox &other);
    void add_power(StringBox &other);
    void enclose_abs();
    void enclose_parens();
    void enclose_sqbrackets();
    void enclose_curlies();
    void enclose_floor();
    void enclose_ceiling();
    void enclose_sqrt();
    void add_right(StringBox &other);
    void add_left_parens();
    void add_right_parens();
    void add_left_sqbracket();
    void add_right_sqbracket();
    void add_left_curly();
    void add_right_curly();
};

}; // namespace SymEngine
