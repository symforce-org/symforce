#include <symforce/examples/robot_3d_localization/run_dynamic_size.h>

int main(int argc, char** argv) {
  // This SYM_ASSERTs on failure instead of CHECK, since it isn't a test
  robot_3d_localization::RunDynamic();
}
