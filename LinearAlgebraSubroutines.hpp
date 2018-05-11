#ifndef RKTK_LINEAR_ALGEBRA_SUBROUTINES_HPP_INCLUDED
#define RKTK_LINEAR_ALGEBRA_SUBROUTINES_HPP_INCLUDED

// C++ standard library headers
#include <cmath> // for std::sqrt
#include <cstddef> // for std::size_t

template <typename T>
T dot(std::size_t n, const T *v, const T *w) {
    T result = v[0] * w[0];
    for (std::size_t i = 1; i < n; ++i) { result += v[i] * w[i]; }
    return result;
}

template <typename T>
bool elementwise_equal(std::size_t n, const T *v, const T *w) {
    for (std::size_t i = 0; i < n; ++i) {
        if (v[i] != w[i]) { return false; }
    }
    return true;
}

template <typename T>
void identity_matrix(T *mat, std::size_t n) {
    const T zero(0), one(1);
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j, ++k) {
            mat[k] = (i == j) ? one : zero;
        }
    }
}

template <typename T>
T euclidean_norm(std::size_t n, const T *v) {
    using std::sqrt;
    T result = v[0] * v[0];
    for (std::size_t i = 1; i < n; ++i) { result += v[i] * v[i]; }
    return sqrt(result);
}

template <typename T>
void matrix_vector_multiply(T *__restrict__ dst,
                            std::size_t n,
                            const T *__restrict__ mat,
                            const T *__restrict__ vec) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = mat[k] * vec[0];
        ++k;
        for (std::size_t j = 1; j < n; ++j, ++k) { dst[i] += mat[k] * vec[j]; }
    }
}

#endif // RKTK_LINEAR_ALGEBRA_SUBROUTINES_HPP_INCLUDED
