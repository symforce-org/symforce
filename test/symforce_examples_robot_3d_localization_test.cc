/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <symforce/examples/robot_3d_localization/run_dynamic_size.h>

int main() {
  // This SYM_ASSERTs on failure instead of CHECK, since it isn't a test
  robot_3d_localization::RunDynamic();
}
