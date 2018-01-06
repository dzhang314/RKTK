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
#include "objective_function.hpp"
#include "linalg_subroutines.hpp"
#include "bfgs_subroutines.hpp"

static inline bool is_dec_digit(char c) { return ('0' <= c) && (c <= '9'); }

static inline bool is_hex_digit(char c) {
    return ('0' <= c) && (c <= '9')
           || ('a' <= c) && (c <= 'f')
           || ('A' <= c) && (c <= 'F');
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
        x_norm = l2_norm(n, x);
        func = objective_function(x);
        objective_gradient(grad, n, x);
        grad_norm = l2_norm(n, grad);
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
        x_norm = l2_norm(n, x);
        func = objective_function(x);
        objective_gradient(grad, n, x);
        grad_norm = l2_norm(n, grad);
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
        const T rec_step_norm = -1.0 / l2_norm(n, step_dir);
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
            const T grad_step_norm = -1.0 / l2_norm(n, step_dir);
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
        x_new_norm = l2_norm(n, x_new);
        // Evaluate the objective function at the new point.
        func_new = objective_function(x_new);
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
        objective_gradient(grad_new, n, x_new);
        grad_new_norm = l2_norm(n, grad_new);
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

static inline void nan_check(const char *msg) {
    if (mpfr_nanflag_p()) {
        std::cout << "INTERNAL ERROR: Invalid calculation performed "
                  << msg << "." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <>
class BFGSOptimizer<mpfr_t> {

    typedef std::numeric_limits<std::uint_fast64_t> uint_fast64_limits;

private: // ======================================================= DATA MEMBERS

    std::size_t n;
    const mpfr_prec_t prec;
    const mpfr_rnd_t rnd;

    mpfr_t *__restrict__ x = nullptr;
    mpfr_t *__restrict__ x_new = nullptr;
    mpfr_t x_norm;
    mpfr_t x_new_norm;

    mpfr_t func;
    mpfr_t func_new;

    mpfr_t *__restrict__ grad = nullptr;
    mpfr_t *__restrict__ grad_new = nullptr;
    mpfr_t grad_norm;
    mpfr_t grad_new_norm;
    mpfr_t *__restrict__ const grad_delta = nullptr;

    mpfr_t step_size;
    mpfr_t step_size_new;
    mpfr_t *__restrict__ const step_dir = nullptr;

    mpfr_t *__restrict__ const hess_inv = nullptr;

    std::size_t iter_count = std::numeric_limits<std::size_t>::max();

    std::uint_fast64_t uuid_seg0 = uint_fast64_limits::max();
    std::uint_fast64_t uuid_seg1 = uint_fast64_limits::max();
    std::uint_fast64_t uuid_seg2 = uint_fast64_limits::max();
    std::uint_fast64_t uuid_seg3 = uint_fast64_limits::max();
    std::uint_fast64_t uuid_seg4 = uint_fast64_limits::max();

public: // ======================================================== CONSTRUCTORS

    explicit BFGSOptimizer(std::size_t num_variables,
                           mpfr_prec_t numeric_precision,
                           mpfr_rnd_t rounding_mode) :
            n(num_variables), prec(numeric_precision), rnd(rounding_mode),
            x(new mpfr_t[n]), x_new(new mpfr_t[n]),
            grad(new mpfr_t[n]), grad_new(new mpfr_t[n]),
            grad_delta(new mpfr_t[n]), step_dir(new mpfr_t[n]),
            hess_inv(new mpfr_t[n * n]) {
        for (std::size_t i = 0; i < n; ++i) mpfr_init2(x[i], prec);
        for (std::size_t i = 0; i < n; ++i) mpfr_init2(x_new[i], prec);
        mpfr_init2(x_norm, prec);
        mpfr_init2(x_new_norm, prec);
        mpfr_init2(func, prec);
        mpfr_init2(func_new, prec);
        for (std::size_t i = 0; i < n; ++i) mpfr_init2(grad[i], prec);
        for (std::size_t i = 0; i < n; ++i) mpfr_init2(grad_new[i], prec);
        mpfr_init2(grad_norm, prec);
        mpfr_init2(grad_new_norm, prec);
        for (std::size_t i = 0; i < n; ++i) mpfr_init2(grad_delta[i], prec);
        mpfr_init2(step_size, prec);
        mpfr_init2(step_size_new, prec);
        for (std::size_t i = 0; i < n; ++i) mpfr_init2(step_dir[i], prec);
        for (std::size_t i = 0; i < n * n; ++i) mpfr_init2(hess_inv[i], prec);
    }

    // explicitly disallow copy construction
    BFGSOptimizer(const BFGSOptimizer &) = delete;

    // explicitly disallow copy assignment
    BFGSOptimizer &operator=(const BFGSOptimizer &) = delete;

public: // ========================================================== DESTRUCTOR

    ~BFGSOptimizer() {
        for (std::size_t i = 0; i < n; ++i) mpfr_clear(x[i]);
        delete[] x;
        for (std::size_t i = 0; i < n; ++i) mpfr_clear(x_new[i]);
        delete[] x_new;
        mpfr_clear(x_norm);
        mpfr_clear(x_new_norm);
        mpfr_clear(func);
        mpfr_clear(func_new);
        for (std::size_t i = 0; i < n; ++i) mpfr_clear(grad[i]);
        delete[] grad;
        for (std::size_t i = 0; i < n; ++i) mpfr_clear(grad_new[i]);
        delete[] grad_new;
        mpfr_clear(grad_norm);
        mpfr_clear(grad_new_norm);
        for (std::size_t i = 0; i < n; ++i) mpfr_clear(grad_delta[i]);
        delete[] grad_delta;
        mpfr_clear(step_size);
        mpfr_clear(step_size_new);
        for (std::size_t i = 0; i < n; ++i) mpfr_clear(step_dir[i]);
        delete[] step_dir;
        for (std::size_t i = 0; i < n * n; ++i) mpfr_clear(hess_inv[i]);
        delete[] hess_inv;
    }

public: // ======================================================== INITIALIZERS

    void initialize_random() {
        nan_check("before workspace initialization");
        std::uint_fast64_t seed[std::mt19937_64::state_size];
        std::random_device seed_source;
        std::generate(std::begin(seed), std::end(seed), std::ref(seed_source));
        std::seed_seq seed_sequence(std::begin(seed), std::end(seed));
        std::mt19937_64 random_engine(seed_sequence);
        std::uniform_real_distribution<long double> unif(0.0L, 1.0L);
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_set_ld(x[i], unif(random_engine), rnd);
        }
        l2_norm(x_norm, n, x, rnd);
        objective_function(func, x, prec, rnd);
        objective_gradient(grad, n, x, prec, rnd);
        l2_norm(grad_norm, n, grad, rnd);
        mpfr_set_zero(step_size, 0);
        identity_matrix(hess_inv, n, rnd);
        iter_count = 0;
        uuid_seg0 = random_engine() & 0xFFFFFFFF;
        uuid_seg1 = random_engine() & 0xFFFF;
        uuid_seg2 = random_engine() & 0xFFFF;
        uuid_seg3 = random_engine() & 0xFFFF;
        uuid_seg4 = random_engine() & 0xFFFFFFFFFFFF;
        nan_check("after workspace initialization");
    }

    void initialize_from_file(const std::string &filename) {
        std::cout << "Opening input file '" << filename << "'..." << std::endl;
        std::FILE *input_file = std::fopen(filename.c_str(), "r");
        if (input_file == nullptr) {
            std::cout << "ERROR: could not open input file." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        std::cout << "Successfully opened input file. Reading..." << std::endl;
        for (std::size_t i = 0; i < n; ++i) {
            if (mpfr_inp_str(x[i], input_file, 10, rnd) == 0) {
                std::cout << "ERROR: Could not read input file entry at index "
                          << i << "." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        std::fclose(input_file);
        std::cout << "Successfully read input file." << std::endl;
        l2_norm(x_norm, n, x, rnd);
        objective_function(func, x, prec, rnd);
        objective_gradient(grad, n, x, prec, rnd);
        l2_norm(grad_norm, n, grad, rnd);
        mpfr_set_zero(step_size, 0);
        identity_matrix(hess_inv, n, rnd);
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

    bool objective_function_has_decreased() {
        return (mpfr_less_p(func_new, func) != 0);
    }

    void print(int print_precision) {
        if (print_precision <= 0) {
            const long double log102 = 0.301029995663981195213738894724493027L;
            print_precision = static_cast<int>(
                    static_cast<long double>(prec) * log102);
            print_precision += 2;
        }
        mpfr_printf(
                "%012zu | %+.*RNe | %+.*RNe | %+.*RNe | %+.*RNe\n", iter_count,
                print_precision, func, print_precision, grad_norm,
                print_precision, step_size, print_precision, x_norm);
    }

    void write_to_file() {
        const long double log102 = 0.301029995663981195213738894724493027L;
        const int print_precision =
                static_cast<int>(static_cast<long double>(prec) * log102) + 2;
        auto f_score = static_cast<int>(
                -100.0L * std::log10(mpfr_get_ld(func, rnd)));
        if (f_score < 0) { f_score = 0; }
        if (f_score > 9999) { f_score = 9999; }
        auto g_score = static_cast<int>(
                -100.0L * std::log10(mpfr_get_ld(grad_norm, rnd)));
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
        std::string filename_string(filename.str());
        std::FILE *output_file = std::fopen(filename_string.c_str(), "w+");
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_fprintf(output_file, "%+.*RNe\n", print_precision, x[i]);
        }
        mpfr_fprintf(output_file, "\n");
        mpfr_fprintf(output_file, "Objective function value: %+.*RNe\n",
                     print_precision, func);
        mpfr_fprintf(output_file, "Objective gradient norm:  %+.*RNe\n",
                     print_precision, grad_norm);
        mpfr_fprintf(output_file, "Most recent step size:    %+.*RNe\n",
                     print_precision, step_size);
        mpfr_fprintf(output_file, "Distance from origin:     %+.*RNe\n",
                     print_precision, x_norm);
        std::fclose(output_file);
    }

public: // ============================================================ MUTATORS

    void set_step_size() {
        mpfr_set_ui(step_size, 1, rnd);
        mpfr_div_2ui(step_size, step_size,
                     static_cast<unsigned long>(prec / 2), rnd);
    }

    void step(int print_precision) {
        nan_check("before performing BFGS iteration");
        // Compute a quasi-Newton step direction by multiplying the approximate
        // inverse Hessian matrix by the gradient vector. Negate the result to
        // obtain a direction of local decrease (rather than increase).
        matrix_vector_multiply(step_dir, n, hess_inv, grad, rnd);
        nan_check("during calculation of BFGS step direction");
        // Normalize the step direction to ensure consistency of step sizes.
        l2_norm(func_new, n, step_dir, rnd);
        mpfr_si_div(func_new, -1, func_new, rnd);
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_mul(step_dir[i], step_dir[i], func_new, rnd);
        }
        nan_check("during normalization of BFGS step direction");
        // Compute a near-optimal step size via quadratic line search.
        quadratic_line_search(step_size_new, x_new, grad_new, n,
                              x, func, step_size, step_dir, prec, rnd);
        nan_check("during quadratic line search");
        after_line_search:
        if (mpfr_zero_p(step_size_new)) {
            print(print_precision);
            std::cout << "NOTICE: Optimal step size reduced to zero. "
                    "Resetting approximate inverse Hessian matrix to the "
                    "identity matrix and re-trying line search." << std::endl;
            identity_matrix(hess_inv, n, rnd);
            matrix_vector_multiply(step_dir, n, hess_inv, grad, rnd);
            l2_norm(func_new, n, step_dir, rnd);
            mpfr_si_div(func_new, -1, func_new, rnd);
            for (std::size_t i = 0; i < n; ++i) {
                mpfr_mul(step_dir[i], step_dir[i], func_new, rnd);
            }
            quadratic_line_search(step_size_new, x_new, grad_new, n,
                                  x, func, step_size, step_dir, prec, rnd);
            if (mpfr_zero_p(step_size_new)) {
                std::cout << "NOTICE: Optimal step size reduced to zero again "
                        "after Hessian reset. BFGS iteration has converged to "
                        "the requested precision." << std::endl;
                for (std::size_t i = 0; i < n; ++i) {
                    mpfr_set(x_new[i], x[i], rnd);
                }
                mpfr_set(x_new_norm, x_norm, rnd);
                mpfr_set(func_new, func, rnd);
                for (std::size_t i = 0; i < n; ++i) {
                    mpfr_set(grad_new[i], grad[i], rnd);
                }
                mpfr_set(grad_new_norm, grad_norm, rnd);
                for (std::size_t i = 0; i < n; ++i) {
                    mpfr_set_zero(grad_delta[i], 0);
                }
                return;
            }
        }
        // Take a step using the computed step direction and step size.
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_fma(x_new[i], step_size_new, step_dir[i], x[i], rnd);
        }
        nan_check("while taking BFGS step");
        l2_norm(x_new_norm, n, x_new, rnd);
        nan_check("while evaluating norm of new point");
        // Evaluate the objective function at the new point.
        objective_function(func_new, x_new, prec, rnd);
        nan_check("during evaluation of objective function at new point");
        // Ensure that the objective function has decreased.
        if (!objective_function_has_decreased()) {
            print(print_precision);
            if (mpfr_less_p(step_size_new, step_size)) {
                std::cout << "NOTICE: BFGS step failed to decrease "
                        "objective function value after decreasing "
                        "step size. Re-trying line search with smaller "
                        "initial step size." << std::endl;
                mpfr_swap(step_size, step_size_new);
                quadratic_line_search(step_size_new, x_new, grad_new, n,
                                      x, func, step_size, step_dir, prec, rnd);
                nan_check("during quadratic line search");
            } else {
                std::cout << "NOTICE: BFGS step failed to decrease objective "
                        "function after increasing step size. Reverting "
                        "to smaller original step size." << std::endl;
                mpfr_set(step_size_new, step_size, rnd);
            }
            goto after_line_search;
        }
        // Evaluate the gradient vector at the new point.
        objective_gradient(grad_new, n, x_new, prec, rnd);
        nan_check("during evaluation of objective gradient at new point");
        l2_norm(grad_new_norm, n, grad_new, rnd);
        nan_check("while evaluating norm of objective gradient");
        // Use difference between previous and current gradient vectors to
        // perform a rank-one update of the approximate inverse Hessian matrix.
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_sub(grad_delta[i], grad_new[i], grad[i], rnd);
        }
        nan_check("while subtracting consecutive gradient vectors");
        update_inverse_hessian(hess_inv, n,
                               grad_delta, step_size_new, step_dir, prec, rnd);
        nan_check("while updating approximate inverse Hessian");
    }

    void shift() {
        std::swap(x, x_new);
        mpfr_set(x_norm, x_new_norm, rnd);
        mpfr_set(func, func_new, rnd);
        std::swap(grad, grad_new);
        mpfr_set(grad_norm, grad_new_norm, rnd);
        mpfr_set(step_size, step_size_new, rnd);
        ++iter_count;
    }

};

#endif // RKTK_NONLINEAR_OPTIMIZERS_HPP
