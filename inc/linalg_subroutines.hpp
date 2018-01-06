#ifndef RKTK_LINALG_SUBROUTINES_HPP
#define RKTK_LINALG_SUBROUTINES_HPP

// C++ standard library headers
#include <cstddef>

// GNU MPFR library header
#include <mpfr.h>

bool elementwise_equal(std::size_t n,
                       const double *__restrict__ v,
                       const double *__restrict__ w);

bool elementwise_equal(std::size_t n, mpfr_t *v, mpfr_t *w);

void identity_matrix(double *mat, std::size_t n);

void identity_matrix(mpfr_t *mat, std::size_t n, mpfr_rnd_t rnd);

double l2_norm(std::size_t n, const double *v);

void l2_norm(mpfr_t dst, std::size_t n, mpfr_t *v, mpfr_rnd_t rnd);

void matrix_vector_multiply(double *__restrict__ dst, std::size_t n,
                            const double *__restrict__ mat,
                            const double *__restrict__ vec);

void matrix_vector_multiply(mpfr_t *dst, std::size_t n,
                            mpfr_t *mat, mpfr_t *vec, mpfr_rnd_t rnd);

#endif // RKTK_LINALG_SUBROUTINES_HPP
