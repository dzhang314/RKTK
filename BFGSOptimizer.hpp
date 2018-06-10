#ifndef DZNL_BFGS_OPTIMIZER_HPP_INCLUDED
#define DZNL_BFGS_OPTIMIZER_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t
#include <functional> // for std::function

// Eigen linear algebra library headers
#include <Eigen/Core>

// Project-specific headers
#include "QuadraticLineSearcher.hpp"

namespace dznl {

    enum class StepType {
        NONE, GRAD, BFGS
    };

    template <typename T>
    class BFGSOptimizer {

    private: // =============================================== MEMBER VARIABLES

        typedef std::function<T(const T *)> objective_function_t;
        typedef std::function<void(T *, const T *)> objective_gradient_t;

        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXT;
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXT;

        const std::size_t n;
        const objective_function_t f;
        const objective_gradient_t g;

        VectorXT x;
        T fx;
        VectorXT gx;
        VectorXT dg;

        T step_size;
        VectorXT step_dir;
        VectorXT grad_dir;
        MatrixXT hess_inv;

        VectorXT temp;
        std::size_t iter_count;
        StepType last_step_type;

    public: // ===================================================== CONSTRUCTOR

        BFGSOptimizer(std::size_t num_dimensions,
                      const objective_function_t &objective_function,
                      const objective_gradient_t &objective_gradient) :
                n(num_dimensions),
                f(objective_function), g(objective_gradient),
                x(n), fx(0), gx(n), dg(n),
                step_size(0), step_dir(n), grad_dir(n), hess_inv(n, n),
                temp(n), iter_count(0), last_step_type(StepType::NONE) {
            hess_inv.setIdentity();
        }

    public: // ======================================================= ACCESSORS

        const VectorXT &get_current_point() { return x; }

        T get_current_objective_value() { return fx; }

        const VectorXT &get_current_gradient() { return gx; }

        std::size_t get_iteration_count() { return iter_count; }

        T get_last_step_size() { return step_size; }

        StepType get_last_step_type() { return last_step_type; }

    public: // ======================================================== MUTATORS

        void set_current_point(const VectorXT &p) {
            x = p;
            fx = f(p.data());
            g(gx.data(), p.data());
        }

        void set_iteration_count(std::size_t k) { iter_count = k; }

        void set_step_size(const T &h) { step_size = h; }

    private: // ==================================== OPTIMIZATION HELPER METHODS

        void reset_hessian() { hess_inv.setIdentity(); }

        void update_inverse_hessian() {
            temp = hess_inv.template selfadjointView<Eigen::Upper>() * dg;
            const T lambda = step_size * dg.dot(step_dir);
            const T sigma = (lambda + dg.dot(temp)) / (lambda * lambda);
            temp -= (step_size * lambda * sigma / 2) * step_dir;
            hess_inv.template selfadjointView<Eigen::Upper>().rankUpdate(
                    temp, step_dir, -step_size / lambda);
        }

        T competitive_line_search() {
            dznl::QuadraticLineSearcher<T> grad_searcher(f, x, grad_dir);
            grad_searcher.search(step_size);
            dznl::QuadraticLineSearcher<T> bfgs_searcher(f, x, step_dir);
            bfgs_searcher.search(step_size);
            if (grad_searcher.get_best_objective_value() <
                bfgs_searcher.get_best_objective_value()) {
                last_step_type = StepType::GRAD;
                step_dir = grad_dir;
                reset_hessian();
                fx = grad_searcher.get_best_objective_value();
                return grad_searcher.get_best_step_size();
            } else {
                last_step_type = StepType::BFGS;
                fx = bfgs_searcher.get_best_objective_value();
                return bfgs_searcher.get_best_step_size();
            }
        }

    public: // ============================================ OPTIMIZATION METHODS

        bool step() {
            grad_dir = -gx.normalized();
            step_dir = -(
                    hess_inv.template selfadjointView<Eigen::Upper>() * gx);
            step_dir.normalize();
            const T new_step_size = competitive_line_search();
            if (new_step_size == 0) { return false; }
            ++iter_count;
            step_size = new_step_size;
            x += step_size * step_dir;
            g(temp.data(), x.data());
            dg = temp - gx;
            gx.swap(temp);
            update_inverse_hessian();
            return true;
        }

    }; // class BFGSOptimizer

} // namespace dznl

#endif // DZNL_BFGS_OPTIMIZER_HPP_INCLUDED
