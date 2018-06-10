#ifndef RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED
#define RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

#include <mpfr.h>

namespace rktk::detail {

    /*
    * The following functions are helper subroutines used in the calculation
    * of Runge-Kutta order conditions. Their names are deliberately short and
    * unreadable because they are intended to be called thousands of times
    * in machine-generated code. Brevity keeps generated file sizes down.
    *
    *     lrs - Lower-triangular matrix Row Sums
    *     elm - ELementwise Multiplication
    *     esq - Elementwise SQuare
    *     dot - DOT product
    *     lvm - Lower-triangular Matrix-Vector multiplication
    *     sqr - Scalar sQuaRe
    *     sri - Set to Reciprocal of (unsigned) Integer
    *     res - RESult (special operation used to evaluate partial
    *                   derivatives of Runge-Kutta order conditions)
    */

    template <typename T>
    void lrs(T *__restrict__ dst, std::size_t n, const T *__restrict__ mat) {
        for (std::size_t i = 0, k = 0; i < n; ++i) {
            dst[i] = mat[k];
            ++k;
            for (std::size_t j = 0; j < i; ++j, ++k) {
                dst[i] += mat[k];
            }
        }
    }

    template <typename T>
    void lrs(T *__restrict__ dst_re, T *__restrict__ dst_du, std::size_t n,
             const T *__restrict__ mat_re, std::size_t mat_di) {
        lrs(dst_re, n, mat_re);
        // TODO: It should be possible to directly calculate the vector index
        // to which the matrix index k corresponds. Low priority because lrs
        // is only called once per objective function call, but would be nice.
        for (std::size_t i = 0, k = 0; i < n; ++i) {
            dst_du[i] = (k == mat_di) ? 1 : 0;
            ++k;
            for (std::size_t j = 0; j < i; ++j, ++k) {
                if (k == mat_di) { dst_du[i] += 1; }
            }
        }
    }

    void lrsm(mpfr_ptr dst, std::size_t dst_size, mpfr_srcptr mat) {
        for (std::size_t i = 0, k = 0; i < dst_size; ++i) {
            mpfr_set(dst + i, mat + k, MPFR_RNDN);
            ++k;
            for (std::size_t j = 0; j < i; ++j, ++k) {
                mpfr_add(dst + i, dst + i, mat + k, MPFR_RNDN);
            }
        }
    }

    void lrsm(mpfr_ptr dst_re, mpfr_ptr dst_du, std::size_t n,
              mpfr_srcptr mat_re, std::size_t mat_di) {
        lrsm(dst_re, n, mat_re);
        for (std::size_t i = 0, k = 0; i < n; ++i) {
            mpfr_set_si(dst_du + i, k == mat_di, MPFR_RNDN);
            ++k;
            for (std::size_t j = 0; j < i; ++j, ++k) {
                mpfr_add_si(dst_du + i, dst_du + i, k == mat_di, MPFR_RNDN);
            }
        }
    }

    template <typename T>
    void elm(T *__restrict__ dst, std::size_t n,
             const T *__restrict__ v, const T *__restrict__ w) {
        for (std::size_t i = 0; i < n; ++i) { dst[i] = v[i] * w[i]; }
    }

    template <typename T>
    void elm(T *__restrict__ dst_re, T *__restrict__ dst_du, std::size_t n,
             const T *__restrict__ v_re, const T *__restrict__ v_du,
             const T *__restrict__ w_re, const T *__restrict__ w_du) {
        elm(dst_re, n, v_re, w_re);
        for (std::size_t i = 0; i < n; ++i) {
            dst_du[i] = v_du[i] * w_re[i] + v_re[i] * w_du[i];
        }
    }

    void elmm(mpfr_ptr dst, std::size_t n, mpfr_srcptr v, mpfr_srcptr w) {
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_mul(dst + i, v + i, w + i, MPFR_RNDN);
        }
    }

    void elmm(mpfr_ptr dst_re, mpfr_ptr dst_du, std::size_t n,
              mpfr_srcptr v_re, mpfr_srcptr v_du,
              mpfr_srcptr w_re, mpfr_srcptr w_du) {
        elmm(dst_re, n, v_re, w_re);
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_fmma(dst_du + i, v_du + i, w_re + i, v_re + i, w_du + i,
                    MPFR_RNDN);
        }
    }

    template <typename T>
    void esq(T *__restrict__ dst, std::size_t n, const T *__restrict__ v) {
        for (std::size_t i = 0; i < n; ++i) { dst[i] = v[i] * v[i]; }
    }

    template <typename T>
    void esq(T *__restrict__ dst_re, T *__restrict__ dst_du, std::size_t n,
             const T *__restrict__ v_re, const T *__restrict__ v_du) {
        esq(dst_re, n, v_re);
        for (std::size_t i = 0; i < n; ++i) {
            dst_du[i] = v_re[i] * v_du[i];
            dst_du[i] += dst_du[i];
        }
    }

    static inline void esqm(mpfr_ptr dst, std::size_t n, mpfr_srcptr v) {
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_sqr(dst + i, v + i, MPFR_RNDN);
        }
    }

    static inline void esqm(mpfr_ptr dst_re, mpfr_ptr dst_du, std::size_t n,
                            mpfr_srcptr v_re, mpfr_srcptr v_du) {
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_mul(dst_du + i, v_re + i, v_du + i, MPFR_RNDN);
            mpfr_mul_2ui(dst_du + i, dst_du + i, 1, MPFR_RNDN);
            mpfr_sqr(dst_re + i, v_re + i, MPFR_RNDN);
        }
    }

    template <typename T>
    T dot(std::size_t n, const T *v, const T *w) {
        if (n == 0) {
            return static_cast<T>(0);
        } else {
            T result = v[0] * w[0];
            for (std::size_t i = 1; i < n; ++i) { result += v[i] * w[i]; }
            return result;
        }
    }

    void dotm(mpfr_ptr dst, std::size_t n, mpfr_srcptr v, mpfr_srcptr w) {
        if (n == 0) {
            mpfr_set_zero(dst, 0);
        } else {
            mpfr_mul(dst, v, w, MPFR_RNDN);
            for (std::size_t i = 1; i < n; ++i) {
                mpfr_fma(dst, v + i, w + i, dst, MPFR_RNDN);
            }
        }
    }

    template <typename T>
    void lvm(T *__restrict__ dst, std::size_t dst_size, std::size_t mat_size,
             const T *__restrict__ mat, const T *__restrict__ vec) {
        std::size_t skip = mat_size - dst_size;
        std::size_t index = skip * (skip + 1) / 2 - 1;
        for (std::size_t i = 0; i < dst_size; ++i, index += skip, ++skip) {
            dst[i] = dot(i + 1, mat + index, vec);
        }
    }

    template <typename T>
    void lvm(T *__restrict__ dst_re, T *__restrict__ dst_du,
             std::size_t dst_size, std::size_t mat_size,
             const T *__restrict__ mat_re, std::size_t mat_di,
             const T *__restrict__ vec_re, const T *__restrict__ vec_du) {
        lvm(dst_re, dst_size, mat_size, mat_re, vec_re);
        std::size_t skip = mat_size - dst_size;
        std::size_t index = skip * (skip + 1) / 2 - 1;
        for (std::size_t i = 0; i < dst_size; ++i, index += skip, ++skip) {
            dst_du[i] = dot(i + 1, mat_re + index, vec_du);
            if (index <= mat_di && mat_di <= index + i) {
                dst_du[i] += vec_re[mat_di - index];
            }
        }
    }

    void lvmm(mpfr_ptr dst, std::size_t dst_size, std::size_t mat_size,
              mpfr_srcptr mat, mpfr_srcptr vec) {
        std::size_t skp = mat_size - dst_size;
        std::size_t idx = skp * (skp + 1) / 2 - 1;
        for (std::size_t i = 0; i < dst_size; ++i, idx += skp, ++skp) {
            dotm(dst + i, i + 1, mat + idx, vec);
        }
    }

    void lvmm(mpfr_ptr dst_re, mpfr_ptr dst_du,
              std::size_t dst_size, std::size_t mat_size,
              mpfr_srcptr mat_re, std::size_t mat_di,
              mpfr_srcptr vec_re, mpfr_srcptr vec_du) {
        lvmm(dst_re, dst_size, mat_size, mat_re, vec_re);
        std::size_t skp = mat_size - dst_size;
        std::size_t idx = skp * (skp + 1) / 2 - 1;
        for (std::size_t i = 0; i < dst_size; ++i, idx += skp, ++skp) {
            dotm(dst_du + i, i + 1, mat_re + idx, vec_du);
            if (idx <= mat_di && mat_di <= idx + i) {
                mpfr_add(dst_du + i, dst_du + i, vec_re + mat_di - idx,
                        MPFR_RNDN);
            }
        }
    }

    template <typename T>
    T sqr(T x) { return x * x; }

    template <typename T>
    T res(std::size_t n, const T *__restrict__ m_re, const T *__restrict__ m_du,
          std::size_t m_offset, const T *__restrict__ x_re, std::size_t x_di,
          std::size_t x_offset, T gamma) {
        const T a = dot(n, m_re + m_offset, x_re + x_offset) - gamma;
        const T b = ((x_offset <= x_di) && (x_di < x_offset + n))
                    ? dot(n, m_du + m_offset, x_re + x_offset) +
                    m_re[m_offset + (x_di - x_offset)]
                    : dot(n, m_du + m_offset, x_re + x_offset);
        const T c = a * b;
        return c + c;
    }

} // namespace rktk::detail

#endif // RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED
