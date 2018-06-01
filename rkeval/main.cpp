// C standard library headers
#include <stddef.h> // for size_t
#include <stdio.h>
#include <stdlib.h> // for exit, strtoll
#include <string.h> // for strlen

// GNU MPFR multiprecision library headers
#include <mpfr.h>

// Project-specific headers
#include "ObjectiveFunctionData.hpp"
#include "ObjectiveFunctionHelpers.hpp"

using namespace rktk::detail;

static mpfr_t r, s, t, u[M_SIZE], v[M_SIZE], w[G_SIZE], x[NUM_VARS];

static inline void initialize_tabs(mpfr_prec_t prec) {
    mpfr_init2(r, prec);
    mpfr_init2(s, prec);
    mpfr_init2(t, prec);
    for (size_t i = 0; i < M_SIZE; ++i) { mpfr_init2(u[i], prec); }
    for (size_t i = 0; i < M_SIZE; ++i) { mpfr_init2(v[i], prec); }
    for (size_t i = 0; i < G_SIZE; ++i) {
        mpfr_init2(w[i], prec);
        mpfr_set_ui(w[i], +1, MPFR_RNDN);
        mpfr_div_ui(w[i], w[i], G[i], MPFR_RNDN);
    }
    for (size_t i = 0; i < NUM_VARS; ++i) { mpfr_init2(x[i], prec); }
}

void objective_function(mpfr_ptr f, mpfr_srcptr x) {
    for (size_t i = 0; i < G_SIZE; ++i) {
        switch (R[i].f) {
            case rkop::LRS:
                lrsm(u[B[i]], A[i], x);
                break;
            case rkop::LVM:
                lvmm(u[B[i]], A[i], NUM_STAGES, x, u[R[i].x]);
                break;
            case rkop::ESQ:
                esqm(u[B[i]], A[i], u[R[i].x]);
                break;
            case rkop::ELM:
                elmm(u[B[i]], A[i], u[R[i].x], u[R[i].y]);
                break;
        }
    }
    mpfr_set_ui(f, 1, MPFR_RNDN);
    for (size_t i = NUM_VARS - NUM_STAGES; i < NUM_VARS; ++i) {
        mpfr_sub(f, f, x + i, MPFR_RNDN);
    }
    mpfr_sqr(f, f, MPFR_RNDN);
    for (size_t i = 0; i < G_SIZE; ++i) {
        resm(f, s, A[i], u[B[i]], x + NUM_VARS - A[i], w[i]);
    }
}

void objective_function_partial(mpfr_ptr g, mpfr_srcptr x, size_t i) {
    for (size_t j = 0; j < G_SIZE; ++j) {
        switch (R[j].f) {
            case rkop::LRS:
                lrss(u[B[j]], v[B[j]], A[j], x, i);
                break;
            case rkop::LVM:
                lvms(u[B[j]], v[B[j]], A[j], NUM_STAGES, x, i,
                     u[R[j].x], v[R[j].x]);
                break;
            case rkop::ESQ:
                esqz(u[B[j]], v[B[j]], A[j], u[R[j].x], v[R[j].x]);
                break;
            case rkop::ELM:
                elmz(u[B[j]], v[B[j]], A[j],
                     u[R[j].x], v[R[j].x], u[R[j].y], v[R[j].y]);
                break;

        }
    }
    if (i >= NUM_VARS - NUM_STAGES) {
        mpfr_set_si(g, -1, MPFR_RNDN);
        for (size_t j = NUM_VARS - NUM_STAGES; j < NUM_VARS; ++j) {
            mpfr_add(g, g, x + j, MPFR_RNDN);
        }
        mpfr_mul_2ui(g, g, 1, MPFR_RNDN);
    } else {
        mpfr_set_zero(g, 0);
    }
    for (size_t j = 0; j < G_SIZE; ++j) {
        ress(g, s, t, A[j], u[B[j]], v[B[j]], x, i, NUM_VARS - A[j], w[j]);
    }
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
    initialize_tabs(prec);
    fprintf(stderr, " Done.\n");
    read_input_file(argv[2]);
    fprintf(stderr, "Successfully read input file.\n");
    if (argc == 3) {
        objective_function(r, *x);
        mpfr_out_str(stdout, 10, 0, r, MPFR_RNDN);
        putchar('\n');
    } else {
        for (size_t i = 0; i < NUM_VARS; ++i) {
            objective_function_partial(r, *x, i);
            mpfr_out_str(stdout, 10, 0, r, MPFR_RNDN);
            putchar('\n');
        }
    }
}
