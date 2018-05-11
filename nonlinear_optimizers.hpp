#ifndef RKTK_NONLINEAR_OPTIMIZERS_HPP
#define RKTK_NONLINEAR_OPTIMIZERS_HPP

// C++ standard library headers
#include <algorithm>  // for std::generate
#include <cstdint>    // for std::uint_fast64_t
#include <cstdlib>    // for std::exit
#include <fstream>    // for std::ifstream, std::ofstream
#include <functional> // for std::ref
#include <iomanip>    // for std::setw, std::setfill
#include <ios>        // for std::dec, std::hex
#include <iostream>   // for std::cout
#include <iterator>   // for std::begin, std::end
#include <limits>     // for std::numeric_limits
#include <random>     // for std::uniform_real_distribution et al.
#include <sstream>    // for std::ostringstream
#include <string>     // for std::string
#include <utility>    // for std::swap

// RKTK headers
#include "ObjectiveFunction.hpp"
#include "LinearAlgebraSubroutines.hpp"
#include "bfgs_subroutines.hpp"

static inline bool is_dec_digit(char c) { return ('0' <= c) && (c <= '9'); }

static inline bool is_hex_digit(char c) {
    return (('0' <= c) && (c <= '9'))
           || (('a' <= c) && (c <= 'f'))
           || (('A' <= c) && (c <= 'F'));
}

static inline bool is_dec_substr(const std::string &str,
                                 std::string::size_type begin,
                                 std::string::size_type end) {
    for (std::string::size_type i = begin; i < end; ++i) {
        if (!is_dec_digit(str[i])) { return false; }
    }
    return true;
}

static inline bool is_hex_substr(const std::string &str,
                                 std::string::size_type begin,
                                 std::string::size_type end) {
    for (std::string::size_type i = begin; i < end; ++i) {
        if (!is_hex_digit(str[i])) { return false; }
    }
    return true;
}

static inline bool is_rktk_filename(const std::string &filename) {
    if (filename.size() != 68) { return false; }
    if (!is_dec_substr(filename, 0, 4)) { return false; }
    if (filename[4] != '-') { return false; }
    if (!is_dec_substr(filename, 5, 9)) { return false; }
    if (filename[9] != '-') { return false; }
    if (filename[10] != 'R') { return false; }
    if (filename[11] != 'K') { return false; }
    if (filename[12] != 'T') { return false; }
    if (filename[13] != 'K') { return false; }
    if (filename[14] != '-') { return false; }
    if (!is_hex_substr(filename, 15, 23)) { return false; }
    if (filename[23] != '-') { return false; }
    if (!is_hex_substr(filename, 24, 28)) { return false; }
    if (filename[28] != '-') { return false; }
    if (!is_hex_substr(filename, 29, 33)) { return false; }
    if (filename[33] != '-') { return false; }
    if (!is_hex_substr(filename, 34, 38)) { return false; }
    if (filename[38] != '-') { return false; }
    if (!is_hex_substr(filename, 39, 51)) { return false; }
    if (filename[51] != '-') { return false; }
    if (!is_dec_substr(filename, 52, 64)) { return false; }
    if (filename[64] != '.') { return false; }
    if (filename[65] != 't') { return false; }
    if (filename[66] != 'x') { return false; }
    return (filename[67] == 't');
}

static inline std::size_t dec_substr_to_int(const std::string &str,
                                            std::string::size_type begin,
                                            std::string::size_type end) {
    std::istringstream substr_stream(str.substr(begin, end - begin));
    std::size_t result;
    substr_stream >> std::dec >> result;
    return result;
}

static inline std::uint_fast64_t hex_substr_to_int(const std::string &str,
                                                   std::string::size_type begin,
                                                   std::string::size_type end) {
    std::istringstream substr_stream(str.substr(begin, end - begin));
    std::uint_fast64_t result;
    substr_stream >> std::hex >> result;
    return result;
}

template <typename T>
class BFGSOptimizer {

    typedef std::numeric_limits<T> limits;
    typedef std::numeric_limits<std::uint_fast64_t> uint_fast64_limits;

private: // ======================================================= DATA MEMBERS

    std::size_t n;

    T *__restrict__ x = nullptr;
    T *__restrict__ x_new = nullptr;
    T x_norm = limits::signaling_NaN();
    T x_new_norm = limits::signaling_NaN();

    T func = limits::signaling_NaN();
    T func_new = limits::signaling_NaN();

    T *__restrict__ grad = nullptr;
    T *__restrict__ grad_new = nullptr;
    T grad_norm = limits::signaling_NaN();
    T grad_new_norm = limits::signaling_NaN();
    T *__restrict__ const grad_delta = nullptr;

    T step_size = limits::signaling_NaN();
    T step_size_new = limits::signaling_NaN();
    T *__restrict__ const step_dir = nullptr;

    T *__restrict__ const hess_inv = nullptr;

    std::size_t iter_count = std::numeric_limits<std::size_t>::max();

    std::uint_fast64_t uuid_seg0 = uint_fast64_limits::max();
    std::uint_fast64_t uuid_seg1 = uint_fast64_limits::max();
    std::uint_fast64_t uuid_seg2 = uint_fast64_limits::max();
    std::uint_fast64_t uuid_seg3 = uint_fast64_limits::max();
    std::uint_fast64_t uuid_seg4 = uint_fast64_limits::max();

public: // ======================================================== CONSTRUCTORS

    explicit BFGSOptimizer(std::size_t num_variables) :
            n(num_variables), x(new T[n]), x_new(new T[n]),
            grad(new T[n]), grad_new(new T[n]),
            grad_delta(new T[n]), step_dir(new T[n]), hess_inv(new T[n * n]) {}

    // explicitly disallow copy construction
    BFGSOptimizer(const BFGSOptimizer &) = delete;

    // explicitly disallow copy assignment
    BFGSOptimizer &operator=(const BFGSOptimizer &) = delete;

public: // ========================================================== DESTRUCTOR

    ~BFGSOptimizer() {
        delete[] x;
        delete[] x_new;
        delete[] grad;
        delete[] grad_new;
        delete[] grad_delta;
        delete[] step_dir;
        delete[] hess_inv;
    }

public: // ======================================================== INITIALIZERS

    void initialize_random() {
        std::uint_fast64_t seed[std::mt19937_64::state_size];
        std::random_device seed_source;
        std::generate(std::begin(seed), std::end(seed), std::ref(seed_source));
        std::seed_seq seed_sequence(std::begin(seed), std::end(seed));
        std::mt19937_64 random_engine(seed_sequence);
        std::uniform_real_distribution<T> unif(T(0), T(1));
        for (std::size_t i = 0; i < n; ++i) { x[i] = unif(random_engine); }
        x_norm = euclidean_norm(n, x);
        func = rktk::objective_function(x);
        rktk::objective_gradient(grad, x);
        grad_norm = euclidean_norm(n, grad);
        step_size = T(0);
        identity_matrix(hess_inv, n);
        iter_count = 0;
        uuid_seg0 = random_engine() & 0xFFFFFFFF;
        uuid_seg1 = random_engine() & 0xFFFF;
        uuid_seg2 = random_engine() & 0xFFFF;
        uuid_seg3 = random_engine() & 0xFFFF;
        uuid_seg4 = random_engine() & 0xFFFFFFFFFFFF;
    }

    void initialize_from_file(const std::string &filename) {
        std::cout << "Opening input file '" << filename << "'..." << std::endl;
        std::ifstream input_file(filename);
        if (!input_file.good()) {
            std::cout << "ERROR: could not open input file." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        std::cout << "Successfully opened input file. Reading..." << std::endl;
        for (std::size_t i = 0; i < n; ++i) {
            input_file >> x[i];
            if (input_file.fail()) {
                std::cout << "ERROR: Could not read input file entry at index "
                          << i << "." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        std::cout << "Successfully read input file." << std::endl;
        x_norm = euclidean_norm(n, x);
        func = rktk::objective_function(x);
        rktk::objective_gradient(grad, x);
        grad_norm = euclidean_norm(n, grad);
        step_size = T(0);
        identity_matrix(hess_inv, n);
        if (is_rktk_filename(filename)) {
            iter_count = dec_substr_to_int(filename, 52, 64);
            uuid_seg0 = hex_substr_to_int(filename, 15, 23);
            uuid_seg1 = hex_substr_to_int(filename, 24, 28);
            uuid_seg2 = hex_substr_to_int(filename, 29, 33);
            uuid_seg3 = hex_substr_to_int(filename, 34, 38);
            uuid_seg4 = hex_substr_to_int(filename, 39, 51);
        } else {
            iter_count = 0;
            std::uint_fast64_t seed[std::mt19937_64::state_size];
            std::random_device seed_source;
            std::generate(std::begin(seed), std::end(seed),
                          std::ref(seed_source));
            std::seed_seq seed_sequence(std::begin(seed), std::end(seed));
            std::mt19937_64 random_engine(seed_sequence);
            uuid_seg0 = random_engine() & 0xFFFFFFFF;
            uuid_seg1 = random_engine() & 0xFFFF;
            uuid_seg2 = random_engine() & 0xFFFF;
            uuid_seg3 = random_engine() & 0xFFFF;
            uuid_seg4 = random_engine() & 0xFFFFFFFFFFFF;
        }
    }

public: // =========================================================== ACCESSORS

    std::size_t get_iteration_count() { return iter_count; }

    bool objective_function_has_decreased() { return func_new < func; }

    void print(int print_precision) {
        std::ios_base::fmtflags old_flags = std::cout.flags();
        std::cout << std::setfill('0') << std::setw(12) << iter_count;
        std::cout << std::showpos << std::scientific;
        std::cout << std::setprecision(print_precision > 0
                                       ? print_precision
                                       : limits::max_digits10);
        std::cout << " | " << func << " | " << grad_norm;
        std::cout << " | " << step_size << " | " << x_norm << std::endl;
        std::cout.flags(old_flags);
    }

    void write_to_file() {
        auto f_score = static_cast<int>(T(-100) * std::log10(func));
        if (f_score < 0) { f_score = 0; }
        if (f_score > 9999) { f_score = 9999; }
        auto g_score = static_cast<int>(T(-100) * std::log10(grad_norm));
        if (g_score < 0) { g_score = 0; }
        if (g_score > 9999) { g_score = 9999; }
        std::ostringstream filename;
        filename << std::setfill('0') << std::dec;
        filename << std::setw(4) << f_score << '-';
        filename << std::setw(4) << g_score << "-RKTK-";
        filename << std::hex << std::uppercase;
        filename << std::setw(8) << uuid_seg0 << '-';
        filename << std::setw(4) << uuid_seg1 << '-';
        filename << std::setw(4) << uuid_seg2 << '-';
        filename << std::setw(4) << uuid_seg3 << '-';
        filename << std::setw(12) << uuid_seg4 << '-';
        filename << std::dec;
        filename << std::setw(12) << iter_count << ".txt";
        std::ofstream output_file(filename.str());
        output_file << std::showpos << std::scientific
                    << std::setprecision(limits::max_digits10);
        for (std::size_t i = 0; i < n; ++i) { output_file << x[i] << '\n'; }
        output_file << '\n';
        output_file << "Objective function value: " << func << '\n';
        output_file << "Objective gradient norm:  " << grad_norm << '\n';
        output_file << "Most recent step size:    " << step_size << '\n';
        output_file << "Distance from origin:     " << x_norm << '\n';
    }

public: // ============================================================ MUTATORS

    void set_step_size(T h) { step_size = h; }

    void step(int print_precision) {
        // Compute a quasi-Newton step direction by multiplying the approximate
        // inverse Hessian matrix by the gradient vector. Negate the result to
        // obtain a direction of local decrease (rather than increase).
        matrix_vector_multiply(step_dir, n, hess_inv, grad);
        // Normalize the step direction to ensure consistency of step sizes.
        const T rec_step_norm = -1.0 / euclidean_norm(n, step_dir);
        for (std::size_t i = 0; i < n; ++i) { step_dir[i] *= rec_step_norm; }
        // Compute a near-optimal step size via quadratic line search.
        step_size_new = quadratic_line_search(
                x_new, grad_new, // Used here as temporary workspace.
                n, x, func, step_size, step_dir);
        after_line_search:
        if (step_size_new == 0) {
            print(print_precision);
            std::cout << "NOTICE: Optimal step size reduced to zero. "
                    "Resetting approximate inverse Hessian matrix to the "
                    "identity matrix and re-trying line search." << std::endl;
            identity_matrix(hess_inv, n);
            matrix_vector_multiply(step_dir, n, hess_inv, grad);
            const T grad_step_norm = -1.0 / euclidean_norm(n, step_dir);
            for (std::size_t i = 0; i < n; ++i) {
                step_dir[i] *= grad_step_norm;
            }
            step_size_new = quadratic_line_search(
                    x_new, grad_new, // Used here as temporary workspace.
                    n, x, func, step_size, step_dir);
            if (step_size_new == 0) {
                std::cout << "NOTICE: Optimal step size reduced to zero again "
                        "after Hessian reset. BFGS iteration has converged to "
                        "the requested precision." << std::endl;
                for (std::size_t i = 0; i < n; ++i) { x_new[i] = x[i]; }
                x_new_norm = x_norm;
                func_new = func;
                for (std::size_t i = 0; i < n; ++i) { grad_new[i] = grad[i]; }
                grad_new_norm = grad_norm;
                for (std::size_t i = 0; i < n; ++i) { grad_delta[i] = 0.0; }
                return;
            }
        }
        // Take a step using the computed step direction and step size.
        for (std::size_t i = 0; i < n; ++i) {
            x_new[i] = x[i] + step_size_new * step_dir[i];
        }
        x_new_norm = euclidean_norm(n, x_new);
        // Evaluate the objective function at the new point.
        func_new = rktk::objective_function(x_new);
        // Ensure that the objective function has decreased.
        if (!objective_function_has_decreased()) {
            print(print_precision);
            if (step_size_new < step_size) {
                std::cout << "NOTICE: BFGS step failed to decrease "
                        "objective function value after decreasing "
                        "step size. Re-trying line search with smaller "
                        "initial step size." << std::endl;
                step_size = step_size_new;
                step_size_new = quadratic_line_search(
                        x_new, grad_new,  // Used here as temporary workspace.
                        n, x, func, step_size, step_dir);
            } else {
                std::cout << "NOTICE: BFGS step failed to decrease objective "
                        "function after increasing step size. Reverting "
                        "to smaller original step size." << std::endl;
                step_size_new = step_size;
            }
            goto after_line_search;
        }
        // Evaluate the gradient vector at the new point.
        rktk::objective_gradient(grad_new, x_new);
        grad_new_norm = euclidean_norm(n, grad_new);
        // Use difference between previous and current gradient vectors to
        // perform a rank-one update of the approximate inverse Hessian matrix.
        for (std::size_t i = 0; i < n; ++i) {
            grad_delta[i] = grad_new[i] - grad[i];
        }
        update_inverse_hessian(hess_inv, n,
                               grad_delta, step_size_new, step_dir);
    }

    void shift() {
        std::swap(x, x_new);
        x_norm = x_new_norm;
        func = func_new;
        std::swap(grad, grad_new);
        grad_norm = grad_new_norm;
        step_size = step_size_new;
        ++iter_count;
    }

};

#endif // RKTK_NONLINEAR_OPTIMIZERS_HPP
