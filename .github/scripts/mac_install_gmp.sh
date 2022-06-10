#!/bin/bash

# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

set -ex

if [ "$CIBW_ARCHS_MACOS" = "arm64" ]; then
  mkdir arm-homebrew
  curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip 1 -C arm-homebrew
  response=$(./arm-homebrew/bin/brew fetch --force --bottle-tag=arm64_big_sur gmp | grep "Downloaded to: " | cut -c 16-)
  echo $response
  # parsed=($response)
  ./arm-homebrew/bin/brew install $response
fi
