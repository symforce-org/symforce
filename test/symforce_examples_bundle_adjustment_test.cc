#include <symforce/examples/bundle_adjustment_dynamic_size/symforce_dynamic_size_example.h>
#include <symforce/examples/bundle_adjustment_fixed_size/symforce_fixed_size_example.h>

int main(int argc, char** argv) {
  // These SYM_ASSERT on failure instead of assertTrue, since they aren't tests
  sym::bundle_adjustment_fixed_size::RunFixedBundleAdjustment();
  sym::bundle_adjustment_dynamic_size::RunDynamicBundleAdjustment();
}
