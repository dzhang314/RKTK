#ifndef RKTOOLS_OBJECTIVE_FUNCTION_HPP
#define RKTOOLS_OBJECTIVE_FUNCTION_HPP

// C++ standard library headers
#include <cstddef> // for std::size_t

// GNU MPFR library headers
#include <mpfr.h>

#define NUM_VARS 136

double objective_function(const double *x);

void objective_function(mpfr_t f, mpfr_t *x, mpfr_prec_t p, mpfr_rnd_t r);

double objective_function_partial(const double *x, std::size_t i);

void objective_function_partial(mpfr_t f_du, mpfr_t *x, std::size_t i,
                                mpfr_prec_t p, mpfr_rnd_t r);

void objective_gradient(double *__restrict__ dst, std::size_t n,
                        const double *__restrict__ x);

void objective_gradient(mpfr_t *dst, std::size_t n,
                        mpfr_t *x, mpfr_prec_t prec, mpfr_rnd_t rnd);

#endif // RKTOOLS_OBJECTIVE_FUNCTION_HPP
