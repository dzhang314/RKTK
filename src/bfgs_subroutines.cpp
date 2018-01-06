#include "../inc/bfgs_subroutines.hpp"

// RKTK headers
#include "../inc/objective_function.hpp" // for objective_function
#include "../inc/linalg_subroutines.hpp" // for matrix_vector_multiply

static inline double dot(std::size_t n,
                         const double *__restrict__ v,
                         const double *__restrict__ w) {
    double res = v[0] * w[0];
    for (std::size_t i = 1; i < n; ++i) { res += v[i] * w[i]; }
    return res;
}

static inline void dot(mpfr_t dst, std::size_t n,
                       mpfr_t *v, mpfr_t *w, mpfr_rnd_t rnd) {
    mpfr_mul(dst, v[0], w[0], rnd);
    for (std::size_t i = 1; i < n; ++i) {
        mpfr_fma(dst, v[i], w[i], dst, rnd);
    }
}

double quadratic_line_search(double *__restrict__ temp1,
                             double *__restrict__ temp2, std::size_t n,
                             const double *__restrict__ x, double f,
                             double initial_step_size,
                             const double *__restrict__ step_direction) {
    double step_size = initial_step_size;
    for (std::size_t i = 0; i < n; ++i) {
        temp1[i] = x[i] + step_size * step_direction[i];
    }
    double f1 = objective_function(temp1);
    double f2;
    if (f1 < f) {
        int num_increases = 0;
        while (true) {
            const double double_step_size = 2 * step_size;
            for (std::size_t i = 0; i < n; ++i) {
                temp2[i] = x[i] + double_step_size * step_direction[i];
            }
            f2 = objective_function(temp2);
            if (f2 >= f1) {
                break;
            } else {
                step_size = double_step_size;
                {
                    double *const temp_swap = temp1;
                    temp1 = temp2;
                    temp2 = temp_swap;
                }
                f1 = f2;
                ++num_increases;
                if (num_increases >= 4) { return step_size; }
            }
        }
        const double numer = 4.0 * f1 - f2 - 3.0 * f;
        const double denom = 2.0 * f1 - f2 - f;
        const double optimal_step_size = 0.5 * step_size * numer / denom;
        if (0.0 < optimal_step_size && optimal_step_size < 2.0 * step_size) {
            return optimal_step_size;
        } else {
            return step_size;
        }
    } else {
        while (true) {
            const double half_step_size = 0.5 * step_size;
            for (std::size_t i = 0; i < n; ++i) {
                temp2[i] = x[i] + half_step_size * step_direction[i];
            }
            if (elementwise_equal(n, x, temp2)) { return 0.0; }
            f2 = objective_function(temp2);
            if (f2 < f) {
                break;
            } else {
                step_size = half_step_size;
                if (step_size == 0.0) { return 0.0; }
                {
                    double *const temp_swap = temp1;
                    temp1 = temp2;
                    temp2 = temp_swap;
                }
                f1 = f2;
            }
        }
        const double numer = f1 - 4.0 * f2 + 3.0 * f;
        const double denom = f1 - 2.0 * f2 + f;
        const double optimal_step_size = 0.25 * step_size * numer / denom;
        if (0.0 < optimal_step_size && optimal_step_size < step_size) {
            return optimal_step_size;
        } else {
            return 0.5 * step_size;
        }
    }
}

void quadratic_line_search(mpfr_t optimal_step_size,
                           mpfr_t *temp1, mpfr_t *temp2, std::size_t n,
                           mpfr_t *x, mpfr_t f,
                           mpfr_t initial_step_size, mpfr_t *step_direction,
                           mpfr_prec_t prec, mpfr_rnd_t rnd) {
    static bool initialized = false;
    static mpfr_t step_size, next_step_size, f1, f2, numer, denom;
    if (!initialized) {
        mpfr_init2(step_size, prec);
        mpfr_init2(next_step_size, prec);
        mpfr_init2(f1, prec);
        mpfr_init2(f2, prec);
        mpfr_init2(numer, prec);
        mpfr_init2(denom, prec);
        initialized = true;
    }
    mpfr_set(step_size, initial_step_size, rnd);
    for (std::size_t i = 0; i < n; ++i) {
        mpfr_fma(temp1[i], step_size, step_direction[i], x[i], rnd);
    }
    objective_function(f1, temp1, prec, rnd);
    if (mpfr_less_p(f1, f)) {
        int num_increases = 0;
        while (true) {
            mpfr_mul_2ui(next_step_size, step_size, 1, rnd);
            for (std::size_t i = 0; i < n; ++i) {
                mpfr_fma(temp2[i],
                         next_step_size, step_direction[i], x[i], rnd);
            }
            objective_function(f2, temp2, prec, rnd);
            if (mpfr_greaterequal_p(f2, f1)) {
                break;
            } else {
                mpfr_swap(step_size, next_step_size);
                {
                    mpfr_t *const temp_swap = temp1;
                    temp1 = temp2;
                    temp2 = temp_swap;
                }
                mpfr_swap(f1, f2);
                ++num_increases;
                if (num_increases >= 4) {
                    mpfr_swap(optimal_step_size, step_size);
                    return;
                }
            }
        }

        mpfr_mul_2ui(denom, f1, 1, rnd);    // denom = 2*f1
        mpfr_sub(denom, denom, f2, rnd);    // denom = 2*f1 - f2
        mpfr_sub(denom, denom, f, rnd);     // denom = 2*f1 - f2 - f
        mpfr_mul_2ui(numer, f1, 2, rnd);    // numer = 4*f1
        mpfr_sub(numer, numer, f2, rnd);    // numer = 4*f1 - f2
        mpfr_mul_ui(f1, f, 3, rnd);         // temporarily store f1' = 3*f
        mpfr_sub(numer, numer, f1, rnd);    // numer = 4*f1 - f2 - 3*f
        mpfr_div_2ui(optimal_step_size, step_size, 1, rnd);
        mpfr_mul(optimal_step_size, optimal_step_size, numer, rnd);
        mpfr_div(optimal_step_size, optimal_step_size, denom, rnd);
        mpfr_mul_2ui(f2, step_size, 1, rnd);
        if ((mpfr_sgn(optimal_step_size) <= 0) ||
            !mpfr_less_p(optimal_step_size, f2)) {
            mpfr_swap(optimal_step_size, step_size);
        }
    } else {
        while (true) {
            mpfr_div_2ui(next_step_size, step_size, 1, rnd);
            for (std::size_t i = 0; i < n; ++i) {
                mpfr_fma(temp2[i],
                         next_step_size, step_direction[i], x[i], rnd);
            }
            if (elementwise_equal(n, x, temp2)) {
                mpfr_set_ui(optimal_step_size, 0, rnd);
                return;
            }
            objective_function(f2, temp2, prec, rnd);
            if (mpfr_less_p(f2, f)) {
                break;
            } else {
                mpfr_swap(step_size, next_step_size);
                if (mpfr_zero_p(step_size)) {
                    mpfr_swap(optimal_step_size, step_size);
                    return;
                }
                {
                    mpfr_t *const temp_swap = temp1;
                    temp1 = temp2;
                    temp2 = temp_swap;
                }
                mpfr_swap(f1, f2);
            }
        }
        mpfr_mul_2ui(f2, f2, 1, rnd);
        mpfr_sub(denom, f1, f2, rnd);
        mpfr_add(denom, denom, f, rnd);
        mpfr_mul_2ui(f2, f2, 1, rnd);
        mpfr_sub(numer, f1, f2, rnd);
        mpfr_mul_ui(f1, f, 3, rnd);
        mpfr_add(numer, numer, f1, rnd);
        mpfr_div_2ui(optimal_step_size, step_size, 2, rnd);
        mpfr_mul(optimal_step_size, optimal_step_size, numer, rnd);
        mpfr_div(optimal_step_size, optimal_step_size, denom, rnd);
        if ((mpfr_sgn(optimal_step_size) <= 0) ||
            !mpfr_less_p(optimal_step_size, step_size)) {
            mpfr_div_2ui(optimal_step_size, step_size, 1, rnd);
        }
    }
}

void update_inverse_hessian(double *__restrict__ inv_hess, std::size_t n,
                            const double *__restrict__ delta_gradient,
                            double step_size,
                            const double *__restrict__ step_direction) {
    static double *__restrict__ kappa = nullptr;
    if (kappa == nullptr) { kappa = new double[n]; }
    matrix_vector_multiply(kappa, n, inv_hess, delta_gradient);
    const double theta = dot(n, delta_gradient, kappa);
    const double lambda = step_size * dot(n, delta_gradient, step_direction);
    const double sigma = (lambda + theta) / (lambda * lambda);
    const double beta = step_size * lambda * sigma / 2;
    for (std::size_t i = 0; i < n; ++i) {
        kappa[i] -= beta * step_direction[i];
    }
    const double alpha = -step_size / lambda;
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j, ++k) {
            inv_hess[k] += alpha * (kappa[i] * step_direction[j] +
                                    step_direction[i] * kappa[j]);
        }
    }
}

void update_inverse_hessian(mpfr_t *inv_hess, std::size_t n,
                            mpfr_t *delta_gradient,
                            mpfr_t step_size, mpfr_t *step_direction,
                            mpfr_prec_t prec, mpfr_rnd_t rnd) {
    static mpfr_t *kappa = nullptr;
    static mpfr_t theta, lambda, sigma, beta, alpha;
    if (kappa == nullptr) {
        kappa = new mpfr_t[n];
        for (std::size_t i = 0; i < n; ++i) { mpfr_init2(kappa[i], prec); }
        mpfr_init2(theta, prec);
        mpfr_init2(lambda, prec);
        mpfr_init2(sigma, prec);
        mpfr_init2(beta, prec);
        mpfr_init2(alpha, prec);
    }
    // nan_check("during initialization of inverse hessian update workspace");
    matrix_vector_multiply(kappa, n, inv_hess, delta_gradient, rnd);
    // nan_check("during evaluation of kappa");
    dot(theta, n, delta_gradient, kappa, rnd);
    // nan_check("during evaluation of theta");
    dot(lambda, n, delta_gradient, step_direction, rnd);
    mpfr_mul(lambda, lambda, step_size, rnd);
    // nan_check("during evaluation of lambda");
    mpfr_sqr(beta, lambda, rnd);
    mpfr_add(sigma, lambda, theta, rnd);
    mpfr_div(sigma, sigma, beta, rnd);
    // nan_check("during evaluation of sigma");
    mpfr_mul(beta, step_size, lambda, rnd);
    mpfr_mul(beta, beta, sigma, rnd);
    mpfr_div_2ui(beta, beta, 1, rnd);
    for (std::size_t i = 0; i < n; ++i) {
        mpfr_fms(kappa[i], beta, step_direction[i], kappa[i], rnd);
        mpfr_neg(kappa[i], kappa[i], rnd);
    }
    mpfr_div(alpha, step_size, lambda, rnd);
    mpfr_neg(alpha, alpha, rnd);
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j, ++k) {
            mpfr_mul(beta, kappa[i], step_direction[j], rnd);
            mpfr_fma(beta, step_direction[i], kappa[j], beta, rnd);
            mpfr_fma(inv_hess[k], alpha, beta, inv_hess[k], rnd);
        }
    }
}
