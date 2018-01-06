#!/usr/bin/env bash

# suppressed remarks:
# 981:  evaluation of operands in unspecified order
# 1572: unreliability of floating-point equality
# 2547: dual specification as both system and non-system include directory

ICPC_OPTIMIZATION_FLAGS="-std=c++11 -fast -use-intel-optimized-headers \
-no-inline-max-size -no-inline-max-total-size -no-inline-min-size"

ICPC_WARNING_FLAGS="-Wall -Werror -w3 -wd981,1572,2547"

ICPC_PLATFORM_FLAGS="-march=native -mtune=native"
ICPC_FLAGS="$ICPC_OPTIMIZATION_FLAGS $ICPC_PLATFORM_FLAGS $ICPC_WARNING_FLAGS"
EXE_NAME="rksearch-intel"

icpc -std=c++11 $ICPC_FLAGS -c src/linalg_subroutines.cpp -o obj/linalg_subroutines.o
icpc -std=c++11 $ICPC_FLAGS -c src/objective_function.cpp -o obj/objective_function.o
icpc -std=c++11 $ICPC_FLAGS -c src/bfgs_subroutines.cpp -o obj/bfgs_subroutines.o
icpc -std=c++11 $ICPC_FLAGS src/rksearch_main.cpp obj/*.o -o bin/$EXE_NAME -lmpfr -lgmp
rm obj/*.o

ICPC_PLATFORM_FLAGS="-march=skylake-avx512 -mtune=skylake-avx512"
ICPC_FLAGS="$ICPC_OPTIMIZATION_FLAGS $ICPC_PLATFORM_FLAGS $ICPC_WARNING_FLAGS"
EXE_NAME="rksearch-intel-skylake-avx512"

icpc -std=c++11 $ICPC_FLAGS -c src/linalg_subroutines.cpp -o obj/linalg_subroutines.o
icpc -std=c++11 $ICPC_FLAGS -c src/objective_function.cpp -o obj/objective_function.o
icpc -std=c++11 $ICPC_FLAGS -c src/bfgs_subroutines.cpp -o obj/bfgs_subroutines.o
icpc -std=c++11 $ICPC_FLAGS src/rksearch_main.cpp obj/*.o -o bin/$EXE_NAME -lmpfr -lgmp
rm obj/*.o
