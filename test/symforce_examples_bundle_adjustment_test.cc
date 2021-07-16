#include <symforce/examples/bundle_adjustment/run_bundle_adjustment.h>

int main(int argc, char** argv) {
  // This SYM_ASSERTs on failure instead of CHECK, since it isn't a test
  bundle_adjustment::RunBundleAdjustment();
}
