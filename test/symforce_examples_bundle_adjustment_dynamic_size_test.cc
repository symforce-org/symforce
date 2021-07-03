#include <symforce/examples/bundle_adjustment_dynamic_size/symforce_dynamic_size_example.h>

int main(int argc, char** argv) {
  // This SYM_ASSERTs on failure instead of CHECK, since it isn't a test
  sym::bundle_adjustment_dynamic_size::RunDynamicBundleAdjustment();
}
