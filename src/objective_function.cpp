#include "../inc/objective_function.hpp"

/*
 * The following functions are helper subroutines used in the calculation
 * of Runge-Kutta order conditions. Their names are deliberately short and
 * unreadable because they are intended to be called thousands of times
 * in machine-generated code. Brevity keeps generated file sizes down.
 *
 * Each function has a four-character name, where the first three characters
 * indicate the mathematical operation the function performs, and the last
 * character indicates the data type on which the function operates. The
 * three-character mnemonic operation codes are listed below.
 *
 *     lrs - Lower-triangular matrix Row Sums
 *     lvm - Lower-triangular Matrix-Vector multiplication
 *     elm - ELementwise Multiplication
 *     sqr - scalar SQuaRe
 *     esq - Elementwise SQuare
 *     res - RESult (special operation used to evaluate partial
 *                   derivatives of Runge-Kutta order conditions)
 *     sri - Set to Reciprocal of (unsigned) Integer
 *
 * The one-character data type codes are listed below. Currently, only
 * double-precision and arbitrary-precision objective functions have been
 * implemented; other letters are reserved below for possible future expansion.
 *
 *     f - float
 *     d - double
 *     l - long double
 *     m - arbitrary precision (mpfr_t)
 *     p - indexed float
 *     q - indexed double
 *     r - indexed long double
 *     s - indexed arbitrary precision (mpfr_t)
 *     w - dual float
 *     x - dual double
 *     y - dual long double
 *     z - dual arbitrary precision (mpfr_t)
 */

static inline void lrsd(double *__restrict__ dst, std::size_t n,
                        const double *__restrict__ mat) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = mat[k];
        ++k;
        for (std::size_t j = 0; j < i; ++j, ++k) {
            dst[i] += mat[k];
        }
    }
}

static inline void lrsm(mpfr_t *dst, std::size_t dst_size,
                        mpfr_t *mat, mpfr_rnd_t rnd) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < dst_size; ++i) {
        mpfr_set(dst[i], mat[k], rnd);
        ++k;
        for (std::size_t j = 0; j < i; ++j, ++k) {
            mpfr_add(dst[i], dst[i], mat[k], rnd);
        }
    }
}

static inline void lrsq(double *__restrict__ dst_re,
                        double *__restrict__ dst_du, std::size_t n,
                        const double *__restrict__ mat_re,
                        std::size_t mat_di) {
    lrsd(dst_re, n, mat_re);
    // TODO: It should be possible to directly calculate the vector index i
    // to which the matrix index k corresponds. Low priority because the lrs*
    // family of methods are only called once per objective function call,
    // but would be nice nonetheless.
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        dst_du[i] = (k == mat_di) ? 1.0 : 0.0;
        ++k;
        for (std::size_t j = 0; j < i; ++j, ++k) {
            dst_du[i] += (k == mat_di) ? 1.0 : 0.0;
        }
    }
}

static inline void lrss(mpfr_t *dst_re, mpfr_t *dst_du, std::size_t n,
                        mpfr_t *mat_re, std::size_t mat_di, mpfr_rnd_t rnd) {
    lrsm(dst_re, n, mat_re, rnd);
    // TODO: See lsrq above.
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        mpfr_set_si(dst_du[i], k == mat_di, rnd);
        ++k;
        for (std::size_t j = 0; j < i; ++j, ++k) {
            mpfr_add_si(dst_du[i], dst_du[i], k == mat_di, rnd);
        }
    }
}

static inline void elmd(double *__restrict__ dst, std::size_t n,
                        const double *__restrict__ v,
                        const double *__restrict__ w) {
    for (std::size_t i = 0; i < n; ++i) { dst[i] = v[i] * w[i]; }
}

static inline void elmm(mpfr_t *dst, std::size_t n,
                        mpfr_t *v, mpfr_t *w, mpfr_rnd_t rnd) {
    for (std::size_t i = 0; i < n; ++i) { mpfr_mul(dst[i], v[i], w[i], rnd); }
}

static inline void elmx(double *__restrict__ dst_re,
                        double *__restrict__ dst_du, std::size_t n,
                        const double *__restrict__ v_re,
                        const double *__restrict__ v_du,
                        const double *__restrict__ w_re,
                        const double *__restrict__ w_du) {
    elmd(dst_re, n, v_re, w_re);
    for (std::size_t i = 0; i < n; ++i) {
        dst_du[i] = v_du[i] * w_re[i] + v_re[i] * w_du[i];
    }
}

static inline void elmz(mpfr_t *dst_re, mpfr_t *dst_du, std::size_t n,
                        mpfr_t *v_re, mpfr_t *v_du, mpfr_t *w_re, mpfr_t *w_du,
                        mpfr_rnd_t rnd) {
    elmm(dst_re, n, v_re, w_re, rnd);
    for (std::size_t i = 0; i < n; ++i) {
        // TODO: Use mpfr_fmma here when we upgrade to MPFR 4.
        mpfr_mul(dst_du[i], v_du[i], w_re[i], rnd);
        mpfr_fma(dst_du[i], v_re[i], w_du[i], dst_du[i], rnd);
    }
}

static inline void esqd(double *__restrict__ dst, std::size_t n,
                        const double *__restrict__ v) {
    for (std::size_t i = 0; i < n; ++i) { dst[i] = v[i] * v[i]; }
}

static inline void esqm(mpfr_t *dst, std::size_t n, mpfr_t *v, mpfr_rnd_t rnd) {
    for (std::size_t i = 0; i < n; ++i) { mpfr_sqr(dst[i], v[i], rnd); }
}

static inline void esqx(double *__restrict__ dst_re,
                        double *__restrict__ dst_du, std::size_t n,
                        const double *__restrict__ v_re,
                        const double *__restrict__ v_du) {
    esqd(dst_re, n, v_re);
    for (std::size_t i = 0; i < n; ++i) {
        dst_du[i] = 2.0 * v_re[i] * v_du[i];
    }
}

static inline void esqz(mpfr_t *dst_re, mpfr_t *dst_du, std::size_t n,
                        mpfr_t *v_re, mpfr_t *v_du, mpfr_rnd_t rnd) {
    for (std::size_t i = 0; i < n; ++i) {
        mpfr_mul(dst_du[i], v_re[i], v_du[i], rnd);
        mpfr_mul_2ui(dst_du[i], dst_du[i], 1, rnd);
        mpfr_sqr(dst_re[i], v_re[i], rnd);
    }
}

static inline double dotd(std::size_t n,
                          const double *__restrict__ v,
                          const double *__restrict__ w) {
    double res = v[0] * w[0];
    for (std::size_t i = 1; i < n; ++i) { res += v[i] * w[i]; }
    return res;
}

static inline void dotm(mpfr_t dst, std::size_t n,
                        mpfr_t *v, mpfr_t *w, mpfr_rnd_t rnd) {
    mpfr_mul(dst, v[0], w[0], rnd);
    for (std::size_t i = 1; i < n; ++i) {
        mpfr_fma(dst, v[i], w[i], dst, rnd);
    }
}

static inline void lvmd(double *__restrict__ dst,
                        std::size_t dst_size, std::size_t mat_size,
                        const double *__restrict__ mat,
                        const double *__restrict__ vec) {
    std::size_t skp = mat_size - dst_size;
    std::size_t idx = skp * (skp + 1) / 2 - 1;
    for (std::size_t i = 0; i < dst_size; ++i, idx += skp, ++skp) {
        dst[i] = dotd(i + 1, mat + idx, vec);
    }
}

static inline void lvmm(mpfr_t *dst, std::size_t dst_size, std::size_t mat_size,
                        mpfr_t *mat, mpfr_t *vec, mpfr_rnd_t rnd) {
    std::size_t skp = mat_size - dst_size;
    std::size_t idx = skp * (skp + 1) / 2 - 1;
    for (std::size_t i = 0; i < dst_size; ++i, idx += skp, ++skp) {
        dotm(dst[i], i + 1, mat + idx, vec, rnd);
    }
}

static inline void lvmq(double *__restrict__ dst_re,
                        double *__restrict__ dst_du,
                        std::size_t dst_size, std::size_t mat_size,
                        const double *__restrict__ mat_re, std::size_t mat_di,
                        const double *__restrict__ vec_re,
                        const double *__restrict__ vec_du) {
    lvmd(dst_re, dst_size, mat_size, mat_re, vec_re);
    std::size_t skp = mat_size - dst_size;
    std::size_t idx = skp * (skp + 1) / 2 - 1;
    for (std::size_t i = 0; i < dst_size; ++i, idx += skp, ++skp) {
        dst_du[i] = dotd(i + 1, mat_re + idx, vec_du);
        if (idx <= mat_di && mat_di <= idx + i) {
            dst_du[i] += vec_re[mat_di - idx];
        }
    }
}

static inline void lvms(mpfr_t *dst_re, mpfr_t *dst_du,
                        std::size_t dst_size, std::size_t mat_size,
                        mpfr_t *mat_re, std::size_t mat_di,
                        mpfr_t *vec_re, mpfr_t *vec_du, mpfr_rnd_t rnd) {
    lvmm(dst_re, dst_size, mat_size, mat_re, vec_re, rnd);
    std::size_t skp = mat_size - dst_size;
    std::size_t idx = skp * (skp + 1) / 2 - 1;
    for (std::size_t i = 0; i < dst_size; ++i, idx += skp, ++skp) {
        dotm(dst_du[i], i + 1, mat_re + idx, vec_du, rnd);
        if (idx <= mat_di && mat_di <= idx + i) {
            mpfr_add(dst_du[i], dst_du[i], vec_re[mat_di - idx], rnd);
        }
    }
}

static inline double sqrd(double x) { return x * x; }

static inline void srim(mpfr_t dst, unsigned long int src, mpfr_rnd_t rnd) {
    mpfr_set_ui(dst, src, rnd);
    mpfr_ui_div(dst, 1, dst, rnd);
}

static inline double resq(std::size_t n,
                          const double *__restrict__ m_re,
                          const double *__restrict__ m_du,
                          std::size_t m_offset,
                          const double *__restrict__ x_re, std::size_t x_di,
                          std::size_t x_offset, double gamma) {
    const double a = dotd(n, m_re + m_offset, x_re + x_offset) - gamma;
    const double b = dotd(n, m_du + m_offset, x_re + x_offset) +
                     ((x_offset <= x_di && x_di < x_offset + n)
                      ? m_re[m_offset + (x_di - x_offset)] : 0.0);
    return 2.0 * a * b;
}

static inline void ress(mpfr_t dst, mpfr_t tmp_re, mpfr_t tmp_du, std::size_t n,
                        mpfr_t *m_re, mpfr_t *m_du, std::size_t m_offset,
                        mpfr_t *x_re, std::size_t x_di, std::size_t x_offset,
                        mpfr_t gamma, mpfr_rnd_t rnd) {
    dotm(tmp_re, n, m_re + m_offset, x_re + x_offset, rnd);
    mpfr_sub(tmp_re, tmp_re, gamma, rnd);
    mpfr_mul_2ui(tmp_re, tmp_re, 1, rnd);
    dotm(tmp_du, n, m_du + m_offset, x_re + x_offset, rnd);
    if (x_offset <= x_di && x_di < x_offset + n) {
        mpfr_add(tmp_du, tmp_du, m_re[m_offset + (x_di - x_offset)], rnd);
    }
    mpfr_fma(dst, tmp_re, tmp_du, dst, rnd);
}

static inline void resm(mpfr_t f, mpfr_t tmp, std::size_t n,
                        mpfr_t *m, mpfr_t *x, mpfr_t gamma, mpfr_rnd_t rnd) {
    dotm(tmp, n, m, x, rnd);
    mpfr_sub(tmp, tmp, gamma, rnd);
    mpfr_fma(f, tmp, tmp, f, rnd);
}

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
