#!/usr/bin/env bash

set -e
set -x

GCC_STD_FLAGS="-std=c++17"
GCC_WRN_FLAGS="-Wall -Wextra -pedantic -Werror -pedantic-errors"
GCC_OPT_FLAGS="-O3 -flto=8 -fno-fat-lto-objects -march=native"

GCC_FLAGS="$GCC_STD_FLAGS $GCC_WRN_FLAGS $GCC_OPT_FLAGS"

mkdir -p bin

g++-8 $GCC_FLAGS rksearch-main.cpp -lmpfr -lgmp -o bin/rksearch
g++-8 $GCC_FLAGS rkeval-main.cpp -o bin/rkeval -lmpfr -lgmp
