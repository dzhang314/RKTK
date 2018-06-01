#ifndef RKTK_ORDER_CONDITION_EVALUATOR_HPP_INCLUDED
#define RKTK_ORDER_CONDITION_EVALUATOR_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

// GNU MPFR multiprecision library headers
#include <mpfr.h>

// Project-specific headers
#include "OrderConditionData.hpp"
#include "OrderConditionHelpers.hpp"

namespace rktk {

    using detail::A, detail::B, detail::G, detail::R;

    class OrderConditionEvaluator {

    private: // =============================================== MEMBER VARIABLES

        mpfr_t s, t, u[M_SIZE], v[M_SIZE], w[G_SIZE];

    public: // ===================================================== CONSTRUCTOR

        OrderConditionEvaluator(mpfr_prec_t prec) {
            mpfr_init2(s, prec);
            mpfr_init2(t, prec);
            for (std::size_t i = 0; i < M_SIZE; ++i) { mpfr_init2(u[i], prec); }
            for (std::size_t i = 0; i < M_SIZE; ++i) { mpfr_init2(v[i], prec); }
            for (std::size_t i = 0; i < G_SIZE; ++i) {
                mpfr_init2(w[i], prec);
                mpfr_set_ui(w[i], +1, MPFR_RNDN);
                mpfr_div_ui(w[i], w[i], G[i], MPFR_RNDN);
            }
        }

    public: // ======================================================= BIG THREE

        OrderConditionEvaluator(
                const OrderConditionEvaluator &) = delete;

        OrderConditionEvaluator &operator=(
                const OrderConditionEvaluator &) = delete;	
        
        ~OrderConditionEvaluator() {
            mpfr_clear(s);
            mpfr_clear(t);
            for (std::size_t i = 0; i < M_SIZE; ++i) { mpfr_clear(u[i]); }
            for (std::size_t i = 0; i < M_SIZE; ++i) { mpfr_clear(v[i]); }
            for (std::size_t i = 0; i < G_SIZE; ++i) { mpfr_clear(w[i]); }
        }

    public: // =================================================================

        void objective_function(mpfr_ptr f, mpfr_srcptr x) {
            for (std::size_t i = 0; i < G_SIZE; ++i) {
                switch (R[i].f) {
                    case detail::rkop::LRS:
                        detail::lrsm(u[B[i]], A[i], x);
                        break;
                    case detail::rkop::LVM:
                        detail::lvmm(u[B[i]], A[i], NUM_STAGES, x, u[R[i].x]);
                        break;
                    case detail::rkop::ESQ:
                        detail::esqm(u[B[i]], A[i], u[R[i].x]);
                        break;
                    case detail::rkop::ELM:
                        detail::elmm(u[B[i]], A[i], u[R[i].x], u[R[i].y]);
                        break;
                }
            }
            mpfr_set_ui(f, 1, MPFR_RNDN);
            for (std::size_t i = NUM_VARS - NUM_STAGES; i < NUM_VARS; ++i) {
                mpfr_sub(f, f, x + i, MPFR_RNDN);
            }
            mpfr_sqr(f, f, MPFR_RNDN);
            for (std::size_t i = 0; i < G_SIZE; ++i) {
                dot(s, A[i], u[B[i]], x + NUM_VARS - A[i]);
                mpfr_sub(s, s, w[i], MPFR_RNDN);
                mpfr_fma(f, s, s, f, MPFR_RNDN);
            }
        }

        void objective_function_partial(mpfr_ptr g, mpfr_srcptr x,
                                        std::size_t i) {
            for (std::size_t j = 0; j < G_SIZE; ++j) {
                switch (R[j].f) {
                    case detail::rkop::LRS:
                        detail::lrss(u[B[j]], v[B[j]], A[j], x, i);
                        break;
                    case detail::rkop::LVM:
                        detail::lvms(u[B[j]], v[B[j]], A[j], NUM_STAGES, x, i,
                                     u[R[j].x], v[R[j].x]);
                        break;
                    case detail::rkop::ESQ:
                        detail::esqz(u[B[j]], v[B[j]], A[j],
                                     u[R[j].x], v[R[j].x]);
                        break;
                    case detail::rkop::ELM:
                        detail::elmz(u[B[j]], v[B[j]], A[j],
                                     u[R[j].x], v[R[j].x],
                                     u[R[j].y], v[R[j].y]);
                        break;
                }
            }
            if (i >= NUM_VARS - NUM_STAGES) {
                mpfr_set_si(g, -1, MPFR_RNDN);
                for (std::size_t j = NUM_VARS - NUM_STAGES; j < NUM_VARS; ++j) {
                    mpfr_add(g, g, x + j, MPFR_RNDN);
                }
                mpfr_mul_2ui(g, g, 1, MPFR_RNDN);
            } else {
                mpfr_set_zero(g, 0);
            }
            for (std::size_t j = 0; j < G_SIZE; ++j) {
                dot(s, A[j], u[B[j]], x + NUM_VARS - A[j]);
                mpfr_sub(s, s, w[j], MPFR_RNDN);
                mpfr_mul_2ui(s, s, 1, MPFR_RNDN);
                dot(t, A[j], v[B[j]], x + NUM_VARS - A[j]);
                if (i + A[j] >= NUM_VARS) {
                    mpfr_add(t, t, u[B[j + 1] + i - NUM_VARS], MPFR_RNDN);
                }
                mpfr_fma(g, s, t, g, MPFR_RNDN);
            }
        }

    }; // class OrderConditionEvaluator

} // namespace rktk

#endif // RKTK_ORDER_CONDITION_EVALUATOR_HPP_INCLUDED
