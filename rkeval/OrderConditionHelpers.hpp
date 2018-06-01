#ifndef RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED
#define RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

// GNU MPFR multiprecision library headers
#include <mpfr.h>

namespace rktk::detail {

    static inline void lrsm(mpfr_ptr dst, std::size_t dst_size,
                            mpfr_srcptr mat) {
        for (std::size_t i = 0, k = 0; i < dst_size; ++i) {
            mpfr_set(dst + i, mat + k, MPFR_RNDN);
            ++k;
            for (std::size_t j = 0; j < i; ++j, ++k) {
                mpfr_add(dst + i, dst + i, mat + k, MPFR_RNDN);
            }
        }
    }

    static inline void lrss(mpfr_ptr dst_re, mpfr_ptr dst_du, std::size_t n,
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

    static inline void elmm(mpfr_ptr dst, std::size_t n,
                            mpfr_srcptr v, mpfr_srcptr w) {
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_mul(dst + i, v + i, w + i, MPFR_RNDN);
        }
    }

    static inline void elmz(mpfr_ptr dst_re, mpfr_ptr dst_du, std::size_t n,
                            mpfr_srcptr v_re, mpfr_srcptr v_du,
                            mpfr_srcptr w_re, mpfr_srcptr w_du) {
        elmm(dst_re, n, v_re, w_re);
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_fmma(dst_du + i, v_du + i, w_re + i, v_re + i, w_du + i,
                    MPFR_RNDN);
        }
    }

    static inline void esqm(mpfr_ptr dst, std::size_t n, mpfr_srcptr v) {
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_sqr(dst + i, v + i, MPFR_RNDN);
        }
    }

    static inline void esqz(mpfr_ptr dst_re, mpfr_ptr dst_du, std::size_t n,
                            mpfr_srcptr v_re, mpfr_srcptr v_du) {
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_mul(dst_du + i, v_re + i, v_du + i, MPFR_RNDN);
            mpfr_mul_2ui(dst_du + i, dst_du + i, 1, MPFR_RNDN);
            mpfr_sqr(dst_re + i, v_re + i, MPFR_RNDN);
        }
    }

    static inline void dotm(mpfr_ptr dst, std::size_t n,
                            mpfr_srcptr v, mpfr_srcptr w) {
        mpfr_mul(dst, v, w, MPFR_RNDN);
        for (std::size_t i = 1; i < n; ++i) {
            mpfr_fma(dst, v + i, w + i, dst, MPFR_RNDN);
        }
    }

    static inline void lvmm(mpfr_ptr dst,
                            std::size_t dst_size, std::size_t mat_size,
                            mpfr_srcptr mat, mpfr_srcptr vec) {
        std::size_t skp = mat_size - dst_size;
        std::size_t idx = skp * (skp + 1) / 2 - 1;
        for (std::size_t i = 0; i < dst_size; ++i, idx += skp, ++skp) {
            dotm(dst + i, i + 1, mat + idx, vec);
        }
    }

    static inline void lvms(mpfr_ptr dst_re, mpfr_ptr dst_du,
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

    static inline void srim(mpfr_ptr dst, unsigned long int src) {
        mpfr_set_ui(dst, 1, MPFR_RNDN);
        mpfr_div_ui(dst, dst, src, MPFR_RNDN);
    }

    static inline void resm(mpfr_ptr f, mpfr_ptr tmp, std::size_t n,
                            mpfr_srcptr m, mpfr_srcptr x, mpfr_srcptr gamma) {
        dotm(tmp, n, m, x);
        mpfr_sub(tmp, tmp, gamma, MPFR_RNDN);
        mpfr_fma(f, tmp, tmp, f, MPFR_RNDN);
    }

    static inline void ress(mpfr_ptr dst, mpfr_ptr tmp_re, mpfr_ptr tmp_du,
                            std::size_t n,
                            mpfr_srcptr m_re, mpfr_srcptr m_du,
                            mpfr_srcptr x_re, std::size_t x_di,
                            std::size_t x_offset, mpfr_srcptr gamma) {
        dotm(tmp_re, n, m_re, x_re + x_offset);
        mpfr_sub(tmp_re, tmp_re, gamma, MPFR_RNDN);
        mpfr_mul_2ui(tmp_re, tmp_re, 1, MPFR_RNDN);
        dotm(tmp_du, n, m_du, x_re + x_offset);
        if (x_offset <= x_di && x_di < x_offset + n) {
            mpfr_add(tmp_du, tmp_du, m_re + (x_di - x_offset), MPFR_RNDN);
        }
        mpfr_fma(dst, tmp_re, tmp_du, dst, MPFR_RNDN);
    }

} // namespace rktk::detail

#endif // RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED
