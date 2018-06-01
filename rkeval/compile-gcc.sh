#!/usr/bin/env bash

set -x
set -e

GCC_OPT_FLAGS="-O3 -flto=8 -fno-fat-lto-objects -march=native"
GCC_WRN_FLAGS="-Wall -Wextra -pedantic -Werror"
GCC_FLAGS="-std=c++17 $GCC_OPT_FLAGS $GCC_WRN_FLAGS"

mkdir -p bin

g++-8 $GCC_FLAGS main.cpp -o bin/rkeval -lmpfr -lgmp
