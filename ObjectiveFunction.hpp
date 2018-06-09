#ifndef RKTK_OBJECTIVE_FUNCTION_HPP_INCLUDED
#define RKTK_OBJECTIVE_FUNCTION_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

// Project-specific headers
#include "OrderConditionHelpers.hpp"
#include "rkeval/OrderConditionData.hpp"
#include "rkeval/OrderConditionHelpers.hpp"

namespace rktk {

    template <typename T>
    class OrderConditionEvaluator {

        const int order;
        const std::size_t num_ops;

        const std::size_t num_stages;
        const std::size_t num_vars;
        const std::size_t table_size;
        std::vector<T> m_tab;

    public:

        OrderConditionEvaluator(int order, std::size_t num_stages)
            : order(order), num_ops(detail::NUM_OPS[order]),
              num_stages(num_stages),
              num_vars(num_stages * (num_stages + 1) / 2),
              table_size(num_ops * num_stages -
                         detail::TOTAL_SIZE_DEFICIT[num_ops]),
              m_tab(2 * table_size) {
            if (order <= 0) {
                throw std::invalid_argument(
                    "order supplied to rktk::OrderConditionEvaluator "
                    "constructor must be a positive integer");
            } else if (order > 15) {
                throw std::invalid_argument(
                    "rktk::OrderConditionEvaluator only supports orders "
                    "up to 15");
            }
        }

        std::size_t idx(std::size_t i) {
            return num_stages * i - detail::TOTAL_SIZE_DEFICIT[i];
        }

        T objective_function(const T *x) {
            T *__restrict__ const m = m_tab.data();
            for (std::size_t i = 0, pos = 0; i < num_ops; ++i) {
                const std::size_t n = num_stages - detail::SIZE_DEFICIT[i];
                const detail::rkop_t op = detail::OPCODES[i];
                switch (op.f) {
                    case detail::rkop::LRS: {
                        detail::lrs(m + pos, n, x);
                    } break;
                    case detail::rkop::LVM: {
                        detail::lvm(m + pos, n, num_stages, x, m + idx(op.a));
                    } break;
                    case detail::rkop::ESQ: {
                        detail::esq(m + pos, n, m + idx(op.a));
                    } break;
                    case detail::rkop::ELM: {
                        const std::size_t ad = detail::SIZE_DEFICIT[op.a];
                        const std::size_t bd = detail::SIZE_DEFICIT[op.b];
                        const std::size_t md = std::max(ad, bd);
                        detail::elm(m + pos, n,
                                    m + idx(op.a) + md - ad,
                                    m + idx(op.b) + md - bd);
                    } break;
                }
                pos += n;
            }
            T result(1);
            const T one(1);
            for (std::size_t i = num_vars - num_stages; i < num_vars; ++i) {
                result -= x[i];
            }
            result *= result;
            for (std::size_t i = 0, pos = 0; i < num_ops; ++i) {
                const std::size_t n = num_stages - detail::SIZE_DEFICIT[i];
                result += detail::sqr(detail::dot(n, m + pos,
                        x + num_vars - num_stages + detail::SIZE_DEFICIT[i]) -
                        one / detail::GAMMA[i]);
                pos += n;
            }
            return result;
        }

        T objective_function_partial(const T *x, std::size_t i) {
            T *__restrict__ const m_re = m_tab.data();
            T *__restrict__ const m_du = m_tab.data() + table_size;
            for (std::size_t j = 0, pos = 0; j < num_ops; ++j) {
                const std::size_t n = num_stages - detail::SIZE_DEFICIT[j];
                const detail::rkop_t op = detail::OPCODES[j];
                switch (op.f) {
                    case detail::rkop::LRS: {
                        detail::lrs(m_re + pos, m_du + pos, n, x, i);
                    } break;
                    case detail::rkop::LVM: {
                        detail::lvm(m_re + pos, m_du + pos, n, num_stages, x, i,
                                    m_re + idx(op.a), m_du + idx(op.a));
                    } break;
                    case detail::rkop::ESQ: {
                        detail::esq(m_re + pos, m_du + pos, n,
                                    m_re + idx(op.a), m_du + idx(op.a));
                    } break;
                    case detail::rkop::ELM: {
                        const std::size_t ad = detail::SIZE_DEFICIT[op.a];
                        const std::size_t bd = detail::SIZE_DEFICIT[op.b];
                        const std::size_t md = std::max(ad, bd);
                        const std::size_t ao = idx(op.a) + md - ad;
                        const std::size_t bo = idx(op.b) + md - bd;
                        detail::elm(m_re + pos, m_du + pos, n,
                                    m_re + ao, m_du + ao, m_re + bo, m_du + bo);
                    } break;
                }
                pos += n;
            }
            T result(0);
            const T one(1);
            if (i >= num_vars - num_stages) {
                result -= 1;
                for (std::size_t j = num_vars - num_stages; j < num_vars; ++j) {
                    result += x[j];
                }
                result += result;
            }
            for (std::size_t j = 0, pos = 0; j < num_ops; ++j) {
                const std::size_t n = num_stages - detail::SIZE_DEFICIT[j];
                result += detail::res(
                        n, m_re, m_du, pos, x, i,
                        num_vars - num_stages + detail::SIZE_DEFICIT[j],
                        one / detail::GAMMA[j]);
                pos += n;
            }
            return result;
        }

        void objective_gradient(T *__restrict__ g, const T *__restrict__ x) {
            for (std::size_t i = 0; i < num_vars; ++i) {
                g[i] = objective_function_partial(x, i);
            }
        }

    }; // class OrderConditionEvaluator

} // namespace rktk

#endif // RKTK_OBJECTIVE_FUNCTION_HPP_INCLUDED
