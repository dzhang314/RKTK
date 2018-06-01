// C standard library headers
#include <stddef.h> // for size_t
#include <stdio.h>  // for fprintf
#include <stdlib.h> // for exit, strtoll
#include <string.h> // for strlen

// GNU MPFR multiprecision library headers
#include <mpfr.h>

// Project-specific headers
#include "OrderConditionEvaluator.hpp"

mpfr_t r, x[NUM_VARS];

static inline void initialize_tabs(mpfr_prec_t prec) {
    mpfr_init2(r, prec);
    for (size_t i = 0; i < NUM_VARS; ++i) { mpfr_init2(x[i], prec); }
}

static inline mp_prec_t get_precision(char *prec_str) {
    char *end;
    const long int precision = strtol(prec_str, &end, 10);
    const int read_whole_arg = (strlen(prec_str) == (size_t) (end - prec_str));
    const int is_positive = (precision > 0);
    if (read_whole_arg && is_positive) {
        fprintf(stderr, "Requested %ld-bit precision.\n", precision);
    } else {
        fprintf(stderr, "ERROR: Could not interpret command-line argument "
                        "'%s' as a positive integer.\n", prec_str);
        exit(EXIT_FAILURE);
    }
    return (mp_prec_t) precision;
}

static inline void read_input_file(char *filename) {
    FILE *input_file = fopen(filename, "r");
    if (input_file == NULL) {
        fprintf(stderr, "ERROR: could not open input file '%s'.\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Successfully opened input file. Reading...\n");
    for (size_t i = 0; i < NUM_VARS; ++i) {
        if (mpfr_inp_str(x[i], input_file, 10, MPFR_RNDN) == 0) {
            fprintf(stderr, "ERROR: Could not read input file entry "
                            "at index %zu.\n", i);
            exit(EXIT_FAILURE);
        }
    }
    fclose(input_file);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s num-bits input-filename [calculate-grad] \n",
                argv[0]);
        exit(EXIT_FAILURE);
    }
    const mp_prec_t prec = get_precision(argv[1]);
    fprintf(stderr, "Allocating arrays...");
    rktk::OrderConditionEvaluator evaluator(prec);
    initialize_tabs(prec);
    fprintf(stderr, " Done.\n");
    read_input_file(argv[2]);
    fprintf(stderr, "Successfully read input file.\n");
    if (argc == 3) {
        evaluator.objective_function(r, *x);
        mpfr_out_str(stdout, 10, 0, r, MPFR_RNDN);
        putchar('\n');
    } else {
        for (size_t i = 0; i < NUM_VARS; ++i) {
            evaluator.objective_function_partial(r, *x, i);
            mpfr_out_str(stdout, 10, 0, r, MPFR_RNDN);
            putchar('\n');
        }
    }
}
