#include <symforce/examples/bundle_adjustment_fixed_size/symforce_fixed_size_example.h>

int main(int argc, char** argv) {
  // This SYM_ASSERTs on failure instead of CHECK, since it isn't a test
  sym::bundle_adjustment_fixed_size::RunFixedBundleAdjustment();
}
