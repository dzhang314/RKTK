#!/usr/bin/env bash

set -x
set -e

GCC_OPT_FLAGS="-O3 -flto=8 -fno-fat-lto-objects -march=native"
GCC_WRN_FLAGS="-Wall -Wextra -pedantic -Werror"
GCC_FLAGS="-std=c99 $GCC_OPT_FLAGS $GCC_WRN_FLAGS"

mkdir -p bin

gcc-8 $GCC_FLAGS main_dynamic.c -o bin/rkeval-dynamic -lmpfr -lgmp
