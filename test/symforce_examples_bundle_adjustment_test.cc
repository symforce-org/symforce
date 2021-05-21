#include <symforce/examples/bundle_adjustment/symforce_dynamic_size_example.h>
#include <symforce/examples/bundle_adjustment/symforce_fixed_size_example.h>

int main(int argc, char** argv) {
  // These SYM_ASSERT on failure instead of assertTrue, since they aren't tests
  sym::RunFixedBundleAdjustment();
  sym::RunDynamicBundleAdjustment();
}
