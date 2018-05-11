#ifndef RKTK_BFGS_SUBROUTINES_HPP_INCLUDED
#define RKTK_BFGS_SUBROUTINES_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

#include "ObjectiveFunction.hpp"

template <typename T>
T quadratic_line_search(T *__restrict__ temp1, T *__restrict__ temp2,
                        std::size_t n,
                        const T *__restrict__ x, T f, T initial_step_size,
                        const T *__restrict__ step_direction) {
    T step_size = initial_step_size;
    for (std::size_t i = 0; i < n; ++i) {
        temp1[i] = x[i] + step_size * step_direction[i];
    }
    T f1 = rktk::objective_function(temp1);
    T f2;
    if (f1 < f) {
        int num_increases = 0;
        while (true) {
            const T double_step_size = step_size + step_size;
            for (std::size_t i = 0; i < n; ++i) {
                temp2[i] = x[i] + double_step_size * step_direction[i];
            }
            f2 = rktk::objective_function(temp2);
            if (f2 >= f1) {
                break;
            } else {
                step_size = double_step_size;
                {
                    T *const temp_swap = temp1;
                    temp1 = temp2;
                    temp2 = temp_swap;
                }
                f1 = f2;
                ++num_increases;
                if (num_increases >= 4) { return step_size; }
            }
        }
        const T numer = 4.0 * f1 - f2 - 3.0 * f;
        const T denom = 2.0 * f1 - f2 - f;
        const T optimal_step_size = step_size * numer / (denom + denom);
        if (0 < optimal_step_size &&
            optimal_step_size < step_size + step_size) {
            return optimal_step_size;
        } else {
            return step_size;
        }
    } else {
        while (true) {
            const T half_step_size = step_size / 2;
            for (std::size_t i = 0; i < n; ++i) {
                temp2[i] = x[i] + half_step_size * step_direction[i];
            }
            if (elementwise_equal(n, x, temp2)) { return 0.0; }
            f2 = rktk::objective_function(temp2);
            if (f2 < f) {
                break;
            } else {
                step_size = half_step_size;
                if (step_size == 0.0) { return 0.0; }
                {
                    T *const temp_swap = temp1;
                    temp1 = temp2;
                    temp2 = temp_swap;
                }
                f1 = f2;
            }
        }
        const T numer = f1 - 4.0 * f2 + 3.0 * f;
        const T denom = f1 - 2.0 * f2 + f;
        const T optimal_step_size = step_size * numer / (4 * denom);
        if (0.0 < optimal_step_size && optimal_step_size < step_size) {
            return optimal_step_size;
        } else {
            return step_size / 2;
        }
    }
}

template <typename T>
void update_inverse_hessian(T *__restrict__ inv_hess, std::size_t n,
                            const T *__restrict__ delta_gradient,
                            T step_size,
                            const T *__restrict__ step_direction) {
    static T *__restrict__ kappa = nullptr;
    if (kappa == nullptr) { kappa = new T[n]; }
    matrix_vector_multiply(kappa, n, inv_hess, delta_gradient);
    const T theta = dot(n, delta_gradient, kappa);
    const T lambda = step_size * dot(n, delta_gradient, step_direction);
    const T sigma = (lambda + theta) / (lambda * lambda);
    const T beta = step_size * lambda * sigma / 2;
    for (std::size_t i = 0; i < n; ++i) {
        kappa[i] -= beta * step_direction[i];
    }
    const T alpha = -step_size / lambda;
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j, ++k) {
            inv_hess[k] += alpha * (kappa[i] * step_direction[j] +
                                    step_direction[i] * kappa[j]);
        }
    }
}

#endif // RKTK_BFGS_SUBROUTINES_HPP_INCLUDED
