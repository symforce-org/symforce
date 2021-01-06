#include <sstream>

#include "../symforce/opt/key.h"

// TODO(hayk): Use the catch unit testing framework (single header).
#define assertTrue(a)                                      \
  if (!(a)) {                                              \
    std::ostringstream o;                                  \
    o << __FILE__ << ":" << __LINE__ << ": Test failure."; \
    throw std::runtime_error(o.str());                     \
  }

void TestPrint() {
  const sym::Key key1('l', -1820881232627931286, 209745344626);
  const sym::Key key2('l', 209745344626, -1947893602558886325);

  std::ostringstream ss;
  ss << key1 << "\n" << key2 << "\n";
  assertTrue(ss.str() ==
             "l_n1820881232627931286_209745344626\nl_209745344626_n1947893602558886325\n");
}

int main(int argc, char** argv) {
  TestPrint();
}
