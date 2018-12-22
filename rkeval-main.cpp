// C++ standard library includes
#include <cstddef> // for std::size_t
#include <cstdio>  // for std::fopen
#include <cstdlib> // for std::exit, std::strtoll
#include <cstring> // for std::strlen
#include <iostream>
#include <vector>

// GNU MPFR multiprecision library headers
#include <mpfr.h>

// Project-specific headers
#include "OrderConditionEvaluatorMPFR.hpp"

static long long int get_positive_integer_argument(char *str) {
    char *end;
    const long long int result = std::strtoll(str, &end, 10);
    const int read_whole_arg = (std::strlen(str) ==
        static_cast<std::size_t>(end - str));
    const int is_positive = (result > 0);
    if (!read_whole_arg || !is_positive) {
        std::cerr << "ERROR: Expected command-line argument '" << str
                  << "' to be a positive integer." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return result;
}

static void read_input_file(mpfr_ptr x, std::size_t n, char *filename) {
    std::FILE *input_file = nullptr;
    if (filename != nullptr) {
        input_file = std::fopen(filename, "r");
        if (input_file == nullptr) {
            std::cerr << "ERROR: Could not open input file '" << filename
                      << "'." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (mpfr_inp_str(x + i, input_file, 10, MPFR_RNDN) == 0) {
            std::cerr << "ERROR: Could not read input file entry at index "
                      << i << "." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    if (input_file != nullptr) { std::fclose(input_file); }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " order num-stages num-bits "
                  << "input-filename [calculate-gradient]" << std::endl;
        return EXIT_FAILURE;
    }
    const unsigned long long int order = static_cast<unsigned long long int>(
        get_positive_integer_argument(argv[1]));
    const std::size_t num_stages = static_cast<std::size_t>(
        get_positive_integer_argument(argv[2]));
    const std::size_t num_vars = num_stages * (num_stages + 1) / 2;
    const mpfr_prec_t prec = static_cast<mp_prec_t>(
        get_positive_integer_argument(argv[3]));
    if (order > 15) {
        std::cerr << "ERROR: Runge-Kutta methods of order greater than 15 "
                  << "are not yet supported." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (order > num_stages) {
        std::cerr << "ERROR: The order of a Runge-Kutta method cannot exceed "
                  << "its number of stages." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    mpfr_t result;
    mpfr_init2(result, prec);
    std::vector<mpfr_t> x(num_vars);
    for (std::size_t i = 0; i < num_vars; ++i) { mpfr_init2(x[i], prec); }
    if (std::strcmp(argv[4], "-") == 0) {
        read_input_file(x[0], num_vars, nullptr);
    } else {
        read_input_file(x[0], num_vars, argv[4]);
    }
    rktk::OrderConditionEvaluatorMPFR evaluator(
        static_cast<int>(order), num_stages, prec);
    if (argc == 5) {
        evaluator.objective_function(result, x[0]);
        mpfr_out_str(stdout, 10, 0, result, MPFR_RNDN);
        std::cout << std::endl;
    } else if (argc == 6) {
        for (std::size_t i = 0; i < num_vars; ++i) {
            evaluator.objective_function_partial(result, x[0], i);
            mpfr_out_str(stdout, 10, 0, result, MPFR_RNDN);
            std::cout << std::endl;
        }
    } else if (argc == 7) {
        evaluator.print_constraint_values(x[0]);
    } else {
        for (std::size_t i = 0; i < num_vars; ++i) {
            evaluator.print_jacobian_values(x[0], i);
        }
    }
}
