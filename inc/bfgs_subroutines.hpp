#ifndef RKTK_BFGS_SUBROUTINES_HPP
#define RKTK_BFGS_SUBROUTINES_HPP

// GNU MPFR library header
#include <mpfr.h>

double quadratic_line_search(double *__restrict__ temp1, double *__restrict__ temp2,
                             std::size_t n, const double *__restrict__ x, double f,
                             double initial_step_size,
                             const double *__restrict__ step_direction);

void quadratic_line_search(mpfr_t optimal_step_size,
                           mpfr_t *temp1, mpfr_t *temp2, std::size_t n,
                           mpfr_t *x, mpfr_t f,
                           mpfr_t initial_step_size,
                           mpfr_t *step_direction,
                           mpfr_prec_t prec, mpfr_rnd_t rnd);

void update_inverse_hessian(double *__restrict__ inv_hess, std::size_t n,
                            const double *__restrict__ delta_gradient,
                            double step_size,
                            const double *__restrict__ step_direction);

void update_inverse_hessian(mpfr_t *inv_hess, std::size_t n,
                            mpfr_t *delta_gradient,
                            mpfr_t step_size, mpfr_t *step_direction,
                            mpfr_prec_t prec, mpfr_rnd_t rnd);

#endif // RKTK_BFGS_SUBROUTINES_HPP
