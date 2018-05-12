#ifndef RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED
#define RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

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
    void lrs(T *__restrict__ dst,
             std::size_t n,
             const T *__restrict__ mat);

    template <typename T>
    void lrs(T *__restrict__ dst_re, T *__restrict__ dst_du,
             std::size_t n,
             const T *__restrict__ mat_re, std::size_t mat_di);

    template <typename T>
    void elm(T *__restrict__ dst,
             std::size_t n,
             const T *__restrict__ v, const T *__restrict__ w);

    template <typename T>
    void elm(T *__restrict__ dst_re, T *__restrict__ dst_du,
             std::size_t n,
             const T *__restrict__ v_re, const T *__restrict__ v_du,
             const T *__restrict__ w_re, const T *__restrict__ w_du);

    template <typename T>
    void esq(T *__restrict__ dst,
             std::size_t n,
             const T *__restrict__ v);

    template <typename T>
    void esq(T *__restrict__ dst_re, T *__restrict__ dst_du,
             std::size_t n,
             const T *__restrict__ v_re, const T *__restrict__ v_du);

    template <typename T>
    T dot(std::size_t n, const T *v, const T *w);

    template <typename T>
    void lvm(T *__restrict__ dst,
             std::size_t dst_size, std::size_t mat_size,
             const T *__restrict__ mat, const T *__restrict__ vec);

    template <typename T>
    void lvm(T *__restrict__ dst_re, T *__restrict__ dst_du,
             std::size_t dst_size, std::size_t mat_size,
             const T *__restrict__ mat_re, std::size_t mat_di,
             const T *__restrict__ vec_re, const T *__restrict__ vec_du);

    template <typename T>
    T sqr(T x);

    template <typename T>
    T res(std::size_t n,
          const T *__restrict__ m_re, const T *__restrict__ m_du,
          std::size_t m_offset,
          const T *__restrict__ x_re, std::size_t x_di,
          std::size_t x_offset, T gamma);

} // namespace rktk::detail

#endif // RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED
