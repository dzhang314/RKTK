#ifndef RKTK_OBJECTIVE_FUNCTION_HPP_INCLUDED
#define RKTK_OBJECTIVE_FUNCTION_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

// Project-specific headers
#include "OrderConditionHelpers.hpp"
#include "rkeval/OrderConditionData.hpp"
#include "rkeval/OrderConditionHelpers.hpp"

#define NUM_VARS 136

static constexpr std::size_t num_ops = 1204;
static constexpr std::size_t num_stages = 16;

namespace rktk {

    static inline std::size_t idx(std::size_t i) {
        using detail::TOTAL_SIZE_DEFICIT;
        return num_stages * i - TOTAL_SIZE_DEFICIT[i];
    }

    template <typename T>
    T objective_function(const T *x) {
        using namespace detail;
        static T *m = nullptr;
        if (m == nullptr) { m = new T[14253]; }
        for (std::size_t i = 0, pos = 0; i < num_ops; ++i) {
            const std::size_t n = num_stages - SIZE_DEFICIT[i];
            const detail::rkop_t op = OPCODES[i];
            switch (op.f) {
                case detail::rkop::LRS: {
                    lrs(m + pos, n, x);
                } break;
                case detail::rkop::LVM: {
                    lvm(m + pos, n, num_stages, x, m + idx(op.a));
                } break;
                case detail::rkop::ESQ: {
                    esq(m + pos, n, m + idx(op.a));
                } break;
                case detail::rkop::ELM: {
                    const std::size_t ad = SIZE_DEFICIT[op.a];
                    const std::size_t bd = SIZE_DEFICIT[op.b];
                    const std::size_t md = std::max(ad, bd);
                    elm(m + pos, n,
                        m + idx(op.a) + md - ad, m + idx(op.b) + md - bd);
                } break;
            }
            pos += n;
        }
        T result(1);
        const T one(1);
        for (std::size_t i = 120; i < 136; ++i) { result -= x[i]; }
        result *= result;
        for (std::size_t i = 0, pos = 0; i < num_ops; ++i) {
            const std::size_t n = num_stages - SIZE_DEFICIT[i];
            result += sqr(dot(n, m + pos, x + 120 + SIZE_DEFICIT[i]) -
                        one / GAMMA[i]);
            pos += n;
        }
        return result;
    }

    template <typename T>
    T objective_function_partial(const T *x, std::size_t i) {
        using namespace detail;
        static T *m_re = nullptr;
        static T *m_du = nullptr;
        if (m_re == nullptr || m_du == nullptr) {
            m_re = new T[14253];
            m_du = new T[14253];
        }
        for (std::size_t j = 0, pos = 0; j < num_ops; ++j) {
            const std::size_t n = num_stages - SIZE_DEFICIT[j];
            const detail::rkop_t op = OPCODES[j];
            switch (op.f) {
                case detail::rkop::LRS: {
                    lrs(m_re + pos, m_du + pos, n, x, i);
                } break;
                case detail::rkop::LVM: {
                    lvm(m_re + pos, m_du + pos, n, num_stages, x, i,
                        m_re + idx(op.a), m_du + idx(op.a));
                } break;
                case detail::rkop::ESQ: {
                    esq(m_re + pos, m_du + pos, n,
                        m_re + idx(op.a), m_du + idx(op.a));
                } break;
                case detail::rkop::ELM: {
                    const std::size_t ad = SIZE_DEFICIT[op.a];
                    const std::size_t bd = SIZE_DEFICIT[op.b];
                    const std::size_t md = std::max(ad, bd);
                    const std::size_t ao = idx(op.a) + md - ad;
                    const std::size_t bo = idx(op.b) + md - bd;
                    elm(m_re + pos, m_du + pos, n,
                        m_re + ao, m_du + ao, m_re + bo, m_du + bo);
                } break;
            }
            pos += n;
        }
        T result(0);
        const T one(1);
        if (i >= 120) {
            result -= 1;
            for (std::size_t j = 120; j < 136; ++j) { result += x[j]; }
            result += result;
        }
        for (std::size_t j = 0, pos = 0; j < num_ops; ++j) {
            const std::size_t n = num_stages - SIZE_DEFICIT[j];
            result += res(n, m_re, m_du, pos, x, i,
                        120 + SIZE_DEFICIT[j], one / GAMMA[j]);
            pos += n;
        }
        return result;
    }

    template <typename T>
    void objective_gradient(T *__restrict__ g, const T *__restrict__ x) {
        for (std::size_t i = 0; i < NUM_VARS; ++i) {
            g[i] = objective_function_partial(x, i);
        }
    }

} // namespace rktk

#endif // RKTK_OBJECTIVE_FUNCTION_HPP_INCLUDED
