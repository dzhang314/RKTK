#include "objective_function.hpp"

#include "OrderConditionHelpers.hpp"
#include "gen/RK_10_16.ipp"

void objective_gradient(double *__restrict__ dst, std::size_t n,
                        const double *__restrict__ x) {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = objective_function_partial(x, i);
    }
}

void objective_gradient(mpfr_t *dst, std::size_t n,
                        mpfr_t *x, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    for (std::size_t i = 0; i < n; ++i) {
        objective_function_partial(dst[i], x, i, prec, rnd);
    }
}
