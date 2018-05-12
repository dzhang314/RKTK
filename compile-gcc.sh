#!/usr/bin/env bash

GCC_STD_FLAGS="-std=c++17"
GCC_WRN_FLAGS="-Wall -Wextra -pedantic -Werror -pedantic-errors"
GCC_OPT_FLAGS="-Ofast -flto=8 -fno-fat-lto-objects -march=native"

GCC_FLAGS="$GCC_STD_FLAGS $GCC_WRN_FLAGS $GCC_OPT_FLAGS"

mkdir -p bin
set -x

if [ ! -f bin/ObjectiveFunction.o ]; then
    g++-7 $GCC_FLAGS -c ObjectiveFunction.cpp -o bin/ObjectiveFunction.o
fi
g++-7 $GCC_FLAGS -c OrderConditionHelpers.cpp -o bin/OrderConditionHelpers.o
g++-7 $GCC_FLAGS -c rksearch_main.cpp -o bin/rksearch_main.o

g++-7 $GCC_OPT_FLAGS \
    bin/ObjectiveFunction.o bin/OrderConditionHelpers.o bin/rksearch_main.o \
    -lmpfr -lgmp -o bin/rksearch
