#include "OrderConditionHelpers.hpp"

#define EXPLICITLY_INSTANTIATE_HELPERS(T) \
    template void rktk::detail::lrs<T>( \
            T *__restrict__ dst, \
            std::size_t n, \
           const T *__restrict__ mat); \
    template void rktk::detail::lrs<T>( \
            T *__restrict__ dst_re, T *__restrict__ dst_du, \
            std::size_t n, \
            const T *__restrict__ mat_re, std::size_t mat_di); \
    template void rktk::detail::elm<T>( \
            T *__restrict__ dst, \
            std::size_t n, \
            const T *__restrict__ v, const T *__restrict__ w); \
    template void rktk::detail::elm<T>( \
            T *__restrict__ dst_re, T *__restrict__ dst_du, \
            std::size_t n, \
            const T *__restrict__ v_re, const T *__restrict__ v_du, \
            const T *__restrict__ w_re, const T *__restrict__ w_du); \
    template void rktk::detail::esq<T>( \
            T *__restrict__ dst, \
            std::size_t n, \
            const T *__restrict__ v); \
    template void rktk::detail::esq<T>( \
            T *__restrict__ dst_re, T *__restrict__ dst_du, \
            std::size_t n, \
            const T *__restrict__ v_re, const T *__restrict__ v_du); \
    template T rktk::detail::dot<T>(std::size_t n, const T *v, const T *w); \
    template void rktk::detail::lvm<T>( \
            T *__restrict__ dst, \
            std::size_t dst_size, std::size_t mat_size, \
            const T *__restrict__ mat, const T *__restrict__ vec); \
    template void rktk::detail::lvm<T>( \
            T *__restrict__ dst_re, T *__restrict__ dst_du, \
            std::size_t dst_size, std::size_t mat_size, \
            const T *__restrict__ mat_re, std::size_t mat_di, \
            const T *__restrict__ vec_re, \
            const T *__restrict__ vec_du); \
    template T rktk::detail::sqr<T>(T x); \
    template T rktk::detail::res<T>( \
            std::size_t n, \
            const T *__restrict__ m_re, const T *__restrict__ m_du, \
            std::size_t m_offset, \
            const T *__restrict__ x_re, std::size_t x_di, \
            std::size_t x_offset, T gamma)


EXPLICITLY_INSTANTIATE_HELPERS(float);

EXPLICITLY_INSTANTIATE_HELPERS(double);

EXPLICITLY_INSTANTIATE_HELPERS(long double);


template <typename T>
void rktk::detail::lrs(T *__restrict__ dst,
                       std::size_t n,
                       const T *__restrict__ mat) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = mat[k];
        ++k;
        for (std::size_t j = 0; j < i; ++j, ++k) {
            dst[i] += mat[k];
        }
    }
}

template <typename T>
void rktk::detail::lrs(T *__restrict__ dst_re, T *__restrict__ dst_du,
                       std::size_t n,
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


template <typename T>
void rktk::detail::elm(T *__restrict__ dst,
                       std::size_t n,
                       const T *__restrict__ v, const T *__restrict__ w) {
    for (std::size_t i = 0; i < n; ++i) { dst[i] = v[i] * w[i]; }
}

template <typename T>
void rktk::detail::elm(T *__restrict__ dst_re, T *__restrict__ dst_du,
                       std::size_t n,
                       const T *__restrict__ v_re, const T *__restrict__ v_du,
                       const T *__restrict__ w_re, const T *__restrict__ w_du) {
    elm(dst_re, n, v_re, w_re);
    for (std::size_t i = 0; i < n; ++i) {
        dst_du[i] = v_du[i] * w_re[i] + v_re[i] * w_du[i];
    }
}


template <typename T>
void rktk::detail::esq(T *__restrict__ dst,
                       std::size_t n,
                       const T *__restrict__ v) {
    for (std::size_t i = 0; i < n; ++i) { dst[i] = v[i] * v[i]; }
}

template <typename T>
void rktk::detail::esq(T *__restrict__ dst_re, T *__restrict__ dst_du,
                       std::size_t n,
                       const T *__restrict__ v_re, const T *__restrict__ v_du) {
    esq(dst_re, n, v_re);
    for (std::size_t i = 0; i < n; ++i) {
        dst_du[i] = v_re[i] * v_du[i];
        dst_du[i] += dst_du[i];
    }
}


template <typename T>
T rktk::detail::dot(std::size_t n, const T *v, const T *w) {
    T result = v[0] * w[0];
    for (std::size_t i = 1; i < n; ++i) { result += v[i] * w[i]; }
    return result;
}


template <typename T>
void rktk::detail::lvm(T *__restrict__ dst,
                       std::size_t dst_size, std::size_t mat_size,
                       const T *__restrict__ mat, const T *__restrict__ vec) {
    std::size_t skip = mat_size - dst_size;
    std::size_t index = skip * (skip + 1) / 2 - 1;
    for (std::size_t i = 0; i < dst_size; ++i, index += skip, ++skip) {
        dst[i] = dot(i + 1, mat + index, vec);
    }
}

template <typename T>
void rktk::detail::lvm(T *__restrict__ dst_re, T *__restrict__ dst_du,
                       std::size_t dst_size, std::size_t mat_size,
                       const T *__restrict__ mat_re, std::size_t mat_di,
                       const T *__restrict__ vec_re,
                       const T *__restrict__ vec_du) {
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


template <typename T>
T rktk::detail::sqr(T x) { return x * x; }


template <typename T>
T rktk::detail::res(std::size_t n,
                    const T *__restrict__ m_re, const T *__restrict__ m_du,
                    std::size_t m_offset,
                    const T *__restrict__ x_re, std::size_t x_di,
                    std::size_t x_offset, T gamma) {
    const T a = dot(n, m_re + m_offset, x_re + x_offset) - gamma;
    const T b = ((x_offset <= x_di) && (x_di < x_offset + n))
                ? dot(n, m_du + m_offset, x_re + x_offset) +
                  m_re[m_offset + (x_di - x_offset)]
                : dot(n, m_du + m_offset, x_re + x_offset);
    const T c = a * b;
    return c + c;
}
