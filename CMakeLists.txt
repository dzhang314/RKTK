cmake_minimum_required(VERSION 3.10)
project(RKTK)

set(CMAKE_CXX_STANDARD 11)

include_directories(inc)

add_compile_options(-O0) # for development

add_executable(RKTK
        OrderConditionHelpers.hpp
        OrderConditionHelpers.cpp
        ObjectiveFunction.hpp
        ObjectiveFunction.cpp
        QuadraticLineSearcher.hpp
        BFGSOptimizer.hpp
        OptimizerDriver.hpp
        rksearch_main.cpp)

target_link_libraries(RKTK mpfr gmp)
