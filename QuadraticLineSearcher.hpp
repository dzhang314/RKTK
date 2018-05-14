#ifndef DZNL_QUADRATIC_LINE_SEARCHER_HPP_INCLUDED
#define DZNL_QUADRATIC_LINE_SEARCHER_HPP_INCLUDED

// C++ standard library headers
#include <functional> // for std::function
#include <stdexcept> // for std::invalid_argument

// Eigen linear algebra library headers
#include <Eigen/Core> // for Eigen::Matrix

namespace dznl {

    template <typename T>
    class QuadraticLineSearcher {

    private: // ========================================== INTERNAL TYPE ALIASES

        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXT;

    private: // =============================================== MEMBER VARIABLES

        const std::size_t n;
        const std::function<T(const T *)> f;

        const VectorXT x0;
        VectorXT xt;
        const VectorXT dx;

        const T f0;
        T f1;
        T f2;

        T best_objective_value;
        T best_step_size;

    public: // ==================================================== CONSTRUCTORS

        explicit QuadraticLineSearcher(
                const std::function<T(const T *)> &objective_function,
                const VectorXT &initial_point, const VectorXT &step_direction)
                : n(static_cast<std::size_t>(initial_point.size())),
                  f(objective_function),
                  x0(initial_point), xt(n), dx(step_direction),
                  f0(objective_function(initial_point.data())), f1(0), f2(0),
                  best_objective_value(f0), best_step_size(0) {
            if (initial_point.size() != step_direction.size()) {
                throw std::invalid_argument(
                        "dznl::QuadraticLineSearcher constructor received "
                        "initial point and step direction vectors of "
                        "different sizes");
            }
        }

    public: // ======================================================= ACCESSORS

        T get_best_objective_value() { return best_objective_value; }

        T get_best_step_size() { return best_step_size; }

    private: // ===================================== LINE SEARCH HELPER METHODS

        T evaluate_objective_function(const T &step_size,
                                      bool *changed = nullptr) {
            xt = x0 + step_size * dx;
            if (x0 == xt) {
                if (changed != nullptr) { *changed = false; }
                return f0;
            } else {
                if (changed != nullptr) { *changed = true; }
            }
            const T objective_value = f(xt.data());
            if (objective_value < best_objective_value) {
                best_objective_value = objective_value;
                best_step_size = step_size;
            }
            return objective_value;
        }

    public: // ============================================= LINE SEARCH METHODS

        void search(T step_size, std::size_t max_increases = 4) {
            f1 = evaluate_objective_function(step_size);
            if (f1 < f0) {
                std::size_t num_increases = 0;
                while (true) {
                    const T double_step_size = step_size + step_size;
                    f2 = evaluate_objective_function(double_step_size);
                    if (f2 >= f1) {
                        break;
                    } else {
                        step_size = double_step_size;
                        f1 = f2;
                        if (++num_increases >= max_increases) { return; }
                    }
                }
                const T numer = 4 * f1 - f2 - 3 * f0;
                const T denom = f1 + f1 - f2 - f0;
                const T optimal_step_size = step_size * numer / (denom + denom);
                evaluate_objective_function(optimal_step_size);
            } else {
                while (true) {
                    const T half_step_size = step_size / 2;
                    bool changed;
                    f2 = evaluate_objective_function(half_step_size, &changed);
                    if (!changed) { return; }
                    if (f2 < f0) {
                        break;
                    } else {
                        step_size = half_step_size;
                        if (step_size == 0) { return; }
                        f1 = f2;
                    }
                }
                const T numer = f1 - 4 * f2 + 3 * f0;
                const T denom = f1 - (f2 + f2) + f0;
                const T optimal_step_size = step_size * numer / (4 * denom);
                evaluate_objective_function(optimal_step_size);
            }
        }

    }; // class QuadraticLineSearcher

} // namespace dznl

#endif // DZNL_QUADRATIC_LINE_SEARCHER_HPP_INCLUDED
