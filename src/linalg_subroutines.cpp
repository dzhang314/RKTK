#include "../inc/linalg_subroutines.hpp"

// C++ standard library headers
#include <cmath> // for std::sqrt

bool elementwise_equal(std::size_t n,
                       const double *__restrict__ v,
                       const double *__restrict__ w) {
    for (std::size_t i = 0; i < n; ++i) {
        if (v[i] != w[i]) { return false; }
    }
    return true;
}

bool elementwise_equal(std::size_t n, mpfr_t *v, mpfr_t *w) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!mpfr_equal_p(v[i], w[i])) { return false; }
    }
    return true;
}

void identity_matrix(double *mat, std::size_t n) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j, ++k) {
            mat[k] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void identity_matrix(mpfr_t *mat, std::size_t n, mpfr_rnd_t rnd) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j, ++k) {
            mpfr_set_si(mat[k], i == j, rnd);
        }
    }
}

double l2_norm(std::size_t n, const double *v) {
    double res = v[0] * v[0];
    for (std::size_t i = 1; i < n; ++i) { res += v[i] * v[i]; }
    return sqrt(res);
}

void l2_norm(mpfr_t dst, std::size_t n, mpfr_t *v, mpfr_rnd_t rnd) {
    mpfr_sqr(dst, v[0], rnd);
    for (std::size_t i = 1; i < n; ++i) { mpfr_fma(dst, v[i], v[i], dst, rnd); }
    mpfr_sqrt(dst, dst, rnd);
}

void matrix_vector_multiply(double *__restrict__ dst, std::size_t n,
                            const double *__restrict__ mat,
                            const double *__restrict__ vec) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = mat[k] * vec[0];
        ++k;
        for (std::size_t j = 1; j < n; ++j, ++k) { dst[i] += mat[k] * vec[j]; }
    }
}

void matrix_vector_multiply(mpfr_t *dst, std::size_t n,
                            mpfr_t *mat, mpfr_t *vec, mpfr_rnd_t rnd) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        mpfr_mul(dst[i], mat[k], vec[0], rnd);
        ++k;
        for (std::size_t j = 1; j < n; ++j, ++k) {
            mpfr_fma(dst[i], mat[k], vec[j], dst[i], rnd);
        }
    }
}
