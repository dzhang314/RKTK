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

// Eigen linear algebra library headers
#include <Eigen/Core>

// RKTK headers
#include "ObjectiveFunction.hpp"
#include "QuadraticLineSearcher.hpp"

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

    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXT;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXT;

    std::size_t n;

    VectorXT x1;
    VectorXT x2;
    T x1_norm;
    T x2_norm;

    T func;
    T func_new;

    VectorXT g1;
    VectorXT g2;
    VectorXT dg;

    T g1_norm;
    T g2_norm;

    T step_size;
    T step_size_new;
    VectorXT step_dir;
    VectorXT grad_dir;

    VectorXT kappa;
    MatrixXT hess_inv;

    std::size_t iter_count;

    std::uint_fast64_t uuid_seg0;
    std::uint_fast64_t uuid_seg1;
    std::uint_fast64_t uuid_seg2;
    std::uint_fast64_t uuid_seg3;
    std::uint_fast64_t uuid_seg4;

public: // ======================================================== CONSTRUCTORS

    explicit BFGSOptimizer(std::size_t num_variables) :
            n(num_variables), x1(n), x2(n), g1(n), g2(n), dg(n),
            step_dir(n), grad_dir(n), kappa(n), hess_inv(n, n) {}

    // explicitly disallow copy construction
    BFGSOptimizer(const BFGSOptimizer &) = delete;

    // explicitly disallow copy assignment
    BFGSOptimizer &operator=(const BFGSOptimizer &) = delete;

public: // ======================================================== INITIALIZERS

    void initialize_random() {
        std::uint_fast64_t seed[std::mt19937_64::state_size];
        std::random_device seed_source;
        std::generate(std::begin(seed), std::end(seed), std::ref(seed_source));
        std::seed_seq seed_sequence(std::begin(seed), std::end(seed));
        std::mt19937_64 random_engine(seed_sequence);
        std::uniform_real_distribution<T> unif(T(0), T(1));
        for (std::size_t i = 0; i < n; ++i) { x1[i] = unif(random_engine); }
        x1_norm = x1.norm();
        func = rktk::objective_function(x1.data());
        rktk::objective_gradient(g1.data(), x1.data());
        g1_norm = g1.norm();
        step_size = T(0);
        hess_inv.setIdentity();
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
            input_file >> x1[i];
            if (input_file.fail()) {
                std::cout << "ERROR: Could not read input file entry at index "
                          << i << "." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        std::cout << "Successfully read input file." << std::endl;
        x1_norm = x1.norm();
        func = rktk::objective_function(x1.data());
        rktk::objective_gradient(g1.data(), x1.data());
        g1_norm = g1.norm();
        step_size = T(0);
        hess_inv.setIdentity();
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
        std::cout << " | " << func << " | " << g1_norm;
        std::cout << " | " << step_size << " | " << x1_norm << std::endl;
        std::cout.flags(old_flags);
    }

    void write_to_file() {
        auto f_score = static_cast<int>(T(-100) * std::log10(func));
        if (f_score < 0) { f_score = 0; }
        if (f_score > 9999) { f_score = 9999; }
        auto g_score = static_cast<int>(T(-100) * std::log10(g1_norm));
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
        for (std::size_t i = 0; i < n; ++i) { output_file << x1[i] << '\n'; }
        output_file << '\n';
        output_file << "Objective function value: " << func << '\n';
        output_file << "Objective gradient norm:  " << g1_norm << '\n';
        output_file << "Most recent step size:    " << step_size << '\n';
        output_file << "Distance from origin:     " << x1_norm << '\n';
    }

public: // ============================================================ MUTATORS

    void set_step_size(T h) { step_size = h; }

    void step(int) {
        static T (*objfun)(const T *) = rktk::objective_function;
        static std::function<T(const T *)> stdfun = objfun;
        // Compute a quasi-Newton step direction by multiplying the approximate
        // inverse Hessian matrix by the gradient vector. Negate the result to
        // obtain a direction of local decrease (rather than increase).
        step_dir = -(hess_inv.template selfadjointView<Eigen::Upper>() * g1);
        // Normalize the step direction to ensure consistency of step sizes.
        step_dir.normalize();
        grad_dir = -g1.normalized();
        // Compute a near-optimal step size via quadratic line search.
        {
            dznl::QuadraticLineSearcher<T> bfgs_searcher(
                    NUM_VARS, stdfun, x1, step_dir);
            bfgs_searcher.search(step_size);
            dznl::QuadraticLineSearcher<T> grad_searcher(
                    NUM_VARS, stdfun, x1, grad_dir);
            grad_searcher.search(step_size);
            if (grad_searcher.get_best_objective_value() <
                bfgs_searcher.get_best_objective_value()) {
                std::cerr << "Gradient" << std::endl;
                step_dir = grad_dir;
                hess_inv.setIdentity();
                step_size_new = grad_searcher.get_best_step_size();
                func_new = grad_searcher.get_best_objective_value();
            } else {
                std::cerr << "BFGS" << std::endl;
                step_size_new = bfgs_searcher.get_best_step_size();
                func_new = bfgs_searcher.get_best_objective_value();
            }
        }
//        if (step_size_new == 0) {
//            print(print_precision);
//            std::cout << "NOTICE: Optimal step size reduced to zero. "
//                         "Resetting approximate inverse Hessian matrix to the "
//                         "identity matrix and re-trying line search."
//                      << std::endl;
//            hess_inv.setIdentity();
//            step_dir = -(hess_inv.template selfadjointView<Eigen::Upper>() * g1);
//            step_dir.normalize();
//            dznl::QuadraticLineSearcher<T> searcher(
//                    NUM_VARS, stdfun, x1, step_dir);
//            searcher.search(step_size);
//            step_size_new = searcher.get_best_step_size();
//            func_new = searcher.get_best_objective_value();
        if (step_size_new == 0) {
            std::cout << "NOTICE: Optimal step size reduced to zero again "
                         "after Hessian reset. BFGS iteration has converged to "
                         "the requested precision." << std::endl;
            x2 = x1;
            x2_norm = x1_norm;
            func_new = func;
            g2 = g1;
            g2_norm = g1_norm;
            dg.setZero();
            return;
        }
//        }
        // Take a step using the computed step direction and step size.
        x2 = x1 + step_size_new * step_dir;
        x2_norm = x2.norm();
        // Evaluate the gradient vector at the new point.
        rktk::objective_gradient(g2.data(), x2.data());
        g2_norm = g2.norm();
        // Use difference between previous and current gradient vectors to
        // perform a rank-one update of the approximate inverse Hessian matrix.
        dg = g2 - g1;
        update_inverse_hessian();
    }

    void update_inverse_hessian() {
        kappa = hess_inv.template selfadjointView<Eigen::Upper>() * dg;
        const T lambda = step_size_new * dg.dot(step_dir);
        const T sigma = (lambda + dg.dot(kappa)) / (lambda * lambda);
        kappa -= (step_size_new * lambda * sigma / 2) * step_dir;
        hess_inv.template selfadjointView<Eigen::Upper>().rankUpdate(
                kappa, step_dir, -step_size_new / lambda);
    }

    void shift() {
        x1.swap(x2);
        x1_norm = x2_norm;
        func = func_new;
        g1.swap(g2);
        g1_norm = g2_norm;
        step_size = step_size_new;
        ++iter_count;
    }

};

#endif // RKTK_NONLINEAR_OPTIMIZERS_HPP
