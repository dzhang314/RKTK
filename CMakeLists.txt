cmake_minimum_required(VERSION 3.10)
project(RKTK)

set(CMAKE_CXX_STANDARD 11)

include_directories(inc)

add_compile_options(-O2 -march=native -mtune=native)

add_executable(RKTK
        inc/bfgs_subroutines.hpp
        inc/linalg_subroutines.hpp
        inc/nonlinear_optimizers.hpp
        inc/objective_function.hpp
        src/bfgs_subroutines.cpp
        src/linalg_subroutines.cpp
        src/objective_function.cpp
        src/rksearch_main.cpp src/OrderConditionHelpers.cpp inc/OrderConditionHelpers.hpp)

target_link_libraries(RKTK mpfr gmp)