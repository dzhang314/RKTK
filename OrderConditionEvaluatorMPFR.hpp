#ifndef RKTK_ORDER_CONDITION_EVALUATOR_MPFR_HPP_INCLUDED
#define RKTK_ORDER_CONDITION_EVALUATOR_MPFR_HPP_INCLUDED

// C++ standard library headers
#include <algorithm> // for std::max
#include <cstddef> // for std::size_t
#include <stdexcept> // for std::invalid_argument
#include <vector>

// GNU MPFR multiprecision library headers
#include <mpfr.h>

// Project-specific headers
#include "OrderConditionData.hpp"
#include "OrderConditionHelpersMPFR.hpp"

namespace rktk {

    class OrderConditionEvaluatorMPFR {

    private: // =============================================== MEMBER VARIABLES

        static constexpr mpfr_rnd_t rnd = MPFR_RNDN;

        const std::size_t num_ops;
        const std::size_t num_stages;
        const std::size_t num_vars;
        const std::size_t table_size;
        mpfr_t r, s, t;
        std::vector<mpfr_t> u, v, w;

    public: // ===================================================== CONSTRUCTOR

        OrderConditionEvaluatorMPFR(int order, std::size_t num_stages,
                                    mpfr_prec_t prec)
            : num_ops(detail::NUM_OPS[order]),
              num_stages(num_stages),
              num_vars(num_stages * (num_stages + 1) / 2),
              table_size(num_ops * num_stages -
                         detail::TOTAL_SIZE_DEFICIT[num_ops]),
              u(table_size), v(table_size), w(num_ops) {
            if (order <= 0) {
                throw std::invalid_argument(
                    "order supplied to rktk::OrderConditionEvaluator "
                    "constructor must be a positive integer");
            } else if (order > 15) {
                throw std::invalid_argument(
                    "rktk::OrderConditionEvaluator only supports orders "
                    "up to 15");
            }
            mpfr_init2(r, prec);
            mpfr_init2(s, prec);
            mpfr_init2(t, prec);
            for (std::size_t i = 0; i < table_size; ++i) {
                mpfr_init2(u[i], prec);
            }
            for (std::size_t i = 0; i < table_size; ++i) {
                mpfr_init2(v[i], prec);
            }
            for (std::size_t i = 0; i < num_ops; ++i) {
                mpfr_init2(w[i], prec);
                mpfr_set_ui(w[i], +1, rnd);
                mpfr_div_ui(w[i], w[i], detail::GAMMA[i], rnd);
            }
        }

    public: // ======================================================= BIG THREE

        OrderConditionEvaluatorMPFR(
                const OrderConditionEvaluatorMPFR &) = delete;

        OrderConditionEvaluatorMPFR &operator=(
                const OrderConditionEvaluatorMPFR &) = delete;

        ~OrderConditionEvaluatorMPFR() {
            mpfr_clear(r);
            mpfr_clear(s);
            mpfr_clear(t);
            for (std::size_t i = 0; i < table_size; ++i) { mpfr_clear(u[i]); }
            for (std::size_t i = 0; i < table_size; ++i) { mpfr_clear(v[i]); }
            for (std::size_t i = 0; i < num_ops;    ++i) { mpfr_clear(w[i]); }
        }

    private: // ================================================================

        std::size_t idx(std::size_t i) {
            return num_stages * i - detail::TOTAL_SIZE_DEFICIT[i];
        }

        void mpfr_println(mpfr_srcptr x) {
            mpfr_out_str(nullptr, 10, 0, x, rnd);
            std::putchar('\n');
            std::fflush(stdout);
        }

        void populate_u(mpfr_srcptr x) {
            for (std::size_t i = 0, pos = 0; i < num_ops; ++i) {
                const std::size_t n = num_stages - detail::SIZE_DEFICIT[i];
                const detail::rkop_t op = detail::OPCODES[i];
                switch (op.f) {
                    case detail::rkop::LRS: {
                        detail::lrsm(u[pos], n, x);
                    } break;
                    case detail::rkop::LVM: {
                        detail::lvmm(u[pos], n, num_stages, x, u[idx(op.a)]);
                    } break;
                    case detail::rkop::ESQ: {
                        detail::esqm(u[pos], n, u[idx(op.a)]);
                    } break;
                    case detail::rkop::ELM: {
                        const std::size_t ad = detail::SIZE_DEFICIT[op.a];
                        const std::size_t bd = detail::SIZE_DEFICIT[op.b];
                        const std::size_t md = std::max(ad, bd);
                        detail::elmm(u[pos], n,
                                     u[idx(op.a) + md - ad],
                                     u[idx(op.b) + md - bd]);
                    } break;
                }
                pos += n;
            }
        }

        void populate_uv(mpfr_srcptr x, std::size_t i) {
            for (std::size_t j = 0, pos = 0; j < num_ops; ++j) {
                const std::size_t n = num_stages - detail::SIZE_DEFICIT[j];
                const detail::rkop_t op = detail::OPCODES[j];
                switch (op.f) {
                    case detail::rkop::LRS: {
                        detail::lrsm(u[pos], v[pos], n, x, i);
                    } break;
                    case detail::rkop::LVM: {
                        detail::lvmm(u[pos], v[pos], n, num_stages, x, i,
                                     u[idx(op.a)], v[idx(op.a)]);
                    } break;
                    case detail::rkop::ESQ: {
                        detail::esqm(u[pos], v[pos], n,
                                     u[idx(op.a)], v[idx(op.a)]);
                    } break;
                    case detail::rkop::ELM: {
                        const std::size_t ad = detail::SIZE_DEFICIT[op.a];
                        const std::size_t bd = detail::SIZE_DEFICIT[op.b];
                        const std::size_t md = std::max(ad, bd);
                        const std::size_t ao = idx(op.a) + md - ad;
                        const std::size_t bo = idx(op.b) + md - bd;
                        detail::elmm(u[pos], v[pos], n,
                                     u[ao], v[ao], u[bo], v[bo]);
                    } break;
                }
                pos += n;
            }
        }

        void populate_v(mpfr_srcptr x, std::size_t i) {
            for (std::size_t j = 0, pos = 0; j < num_ops; ++j) {
                const std::size_t n = num_stages - detail::SIZE_DEFICIT[j];
                const detail::rkop_t op = detail::OPCODES[j];
                switch (op.f) {
                    case detail::rkop::LRS: {
                        detail::lrsm(v[pos], n, i);
                    } break;
                    case detail::rkop::LVM: {
                        detail::lvmm(v[pos], n, num_stages, x, i,
                                     u[idx(op.a)], v[idx(op.a)]);
                    } break;
                    case detail::rkop::ESQ: {
                        detail::esqm(v[pos], n, u[idx(op.a)], v[idx(op.a)]);
                    } break;
                    case detail::rkop::ELM: {
                        const std::size_t ad = detail::SIZE_DEFICIT[op.a];
                        const std::size_t bd = detail::SIZE_DEFICIT[op.b];
                        const std::size_t md = std::max(ad, bd);
                        const std::size_t ao = idx(op.a) + md - ad;
                        const std::size_t bo = idx(op.b) + md - bd;
                        detail::elmm(v[pos], n, u[ao], v[ao], u[bo], v[bo]);
                    } break;
                }
                pos += n;
            }
        }

    public: // =================================================================

        void objective_function(mpfr_ptr f, mpfr_srcptr x) {
            populate_u(x);
            mpfr_set_si(f, -1, rnd);
            for (std::size_t i = num_vars - num_stages; i < num_vars; ++i) {
                mpfr_add(f, f, x + i, rnd);
            }
            mpfr_sqr(f, f, rnd);
            for (std::size_t i = 0, pos = 0; i < num_ops; ++i) {
                const std::size_t n = num_stages - detail::SIZE_DEFICIT[i];
                detail::dotm(s, n, u[pos], x + num_vars - n);
                mpfr_sub(s, s, w[i], rnd);
                mpfr_fma(f, s, s, f, rnd);
                pos += n;
            }
        }

        void print_objective_value(mpfr_srcptr x) {
            populate_u(x);
            mpfr_set_si(t, -1, rnd);
            for (std::size_t i = num_vars - num_stages; i < num_vars; ++i) {
                mpfr_add(t, t, x + i, rnd);
            }
            mpfr_sqr(t, t, rnd);
            for (std::size_t i = 0, pos = 0; i < num_ops; ++i) {
                const std::size_t n = num_stages - detail::SIZE_DEFICIT[i];
                detail::dotm(s, n, u[pos], x + num_vars - n);
                mpfr_sub(s, s, w[i], rnd);
                mpfr_fma(t, s, s, t, rnd);
                pos += n;
            }
            mpfr_println(t);
        }

        void print_constraint_values(mpfr_srcptr x) {
            populate_u(x);
            mpfr_set_si(s, -1, rnd);
            for (std::size_t i = num_vars - num_stages; i < num_vars; ++i) {
                mpfr_add(s, s, x + i, rnd);
            }
            mpfr_println(s);
            for (std::size_t i = 0, pos = 0; i < num_ops; ++i) {
                const std::size_t n = num_stages - detail::SIZE_DEFICIT[i];
                detail::dotm(s, n, u[pos], x + num_vars - n);
                mpfr_sub(s, s, w[i], rnd);
                mpfr_println(s);
                pos += n;
            }
        }

        void objective_function_partial(mpfr_ptr g, mpfr_srcptr x,
                                        std::size_t i) {
            populate_uv(x, i);
            if (i >= num_vars - num_stages) {
                mpfr_set_si(g, -1, rnd);
                for (std::size_t j = num_vars - num_stages; j < num_vars; ++j) {
                    mpfr_add(g, g, x + j, rnd);
                }
                mpfr_mul_2ui(g, g, 1, rnd);
            } else {
                mpfr_set_zero(g, 0);
            }
            for (std::size_t j = 0, pos = 0; j < num_ops; ++j) {
                const std::size_t n = num_stages - detail::SIZE_DEFICIT[j];
                detail::dotm(s, n, u[pos], x + num_vars - n);
                mpfr_sub(s, s, w[j], rnd);
                mpfr_mul_2ui(s, s, 1, rnd);
                detail::dotm(t, n, v[pos], x + num_vars - n);
                if (i + n >= num_vars) {
                    mpfr_add(t, t, u[pos + n + i - num_vars], rnd);
                }
                mpfr_fma(g, s, t, g, rnd);
                pos += n;
            }
        }

        void print_partial_values(mpfr_srcptr x) {
            populate_u(x);
            for (std::size_t i = 0; i < num_vars; ++i) {
                populate_v(x, i);
                if (i >= num_vars - num_stages) {
                    mpfr_set_si(r, -1, rnd);
                    for (std::size_t j = num_vars - num_stages;
                         j < num_vars; ++j) {
                        mpfr_add(r, r, x + j, rnd);
                    }
                    mpfr_mul_2ui(r, r, 1, rnd);
                } else {
                    mpfr_set_zero(r, 0);
                }
                for (std::size_t j = 0, pos = 0; j < num_ops; ++j) {
                    const std::size_t n = num_stages - detail::SIZE_DEFICIT[j];
                    detail::dotm(s, n, u[pos], x + num_vars - n);
                    mpfr_sub(s, s, w[j], rnd);
                    mpfr_mul_2ui(s, s, 1, rnd);
                    detail::dotm(t, n, v[pos], x + num_vars - n);
                    if (i + n >= num_vars) {
                        mpfr_add(t, t, u[pos + n + i - num_vars], rnd);
                    }
                    mpfr_fma(r, s, t, r, rnd);
                    pos += n;
                }
                mpfr_println(r);
            }
        }

        void print_jacobian_values(mpfr_srcptr x) {
            populate_u(x);
            for (std::size_t i = 0; i < num_vars; ++i) {
                populate_v(x, i);
                mpfr_set_si(s, i + num_stages >= num_vars, rnd);
                mpfr_println(s);
                for (std::size_t j = 0, pos = 0; j < num_ops; ++j) {
                    const std::size_t n = num_stages - detail::SIZE_DEFICIT[j];
                    detail::dotm(s, n, v[pos], x + num_vars - n);
                    if (i + n >= num_vars) {
                        mpfr_add(s, s, u[pos + n + i - num_vars], rnd);
                    }
                    mpfr_println(s);
                    pos += n;
                }
            }
        }

    }; // class OrderConditionEvaluatorMPFR

} // namespace rktk

#endif // RKTK_ORDER_CONDITION_EVALUATOR_MPFR_HPP_INCLUDED
