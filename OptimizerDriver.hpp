#ifndef RKTK_OPTIMIZER_DRIVER_HPP_INCLUDED
#define RKTK_OPTIMIZER_DRIVER_HPP_INCLUDED

// C++ standard library headers
#include <algorithm> // for std::generate
#include <chrono>
#include <cstdint> // for std::uint64_t
#include <cstdlib>    // for std::exit
#include <fstream> // for std::ifstream, std::ofstream
#include <functional> // for std::ref
#include <iomanip> // for std::setw, std::setfill
#include <ios> // for std::dec, std::hex
#include <iostream> // for std::cout
#include <limits> // for std::numeric_limits
#include <random> // for std::mt19937 et al.
#include <sstream> // for std::istringstream and std::ostringstream
#include <string>

// Eigen linear algebra library headers
#include <Eigen/Core>

// Project-specific headers
#include "ObjectiveFunction.hpp"
#include "BFGSOptimizer.hpp"

namespace rktk {

    template <typename T>
    class OptimizerDriver {

    private: // ========================================== INTERNAL TYPE ALIASES

        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXT;

        typedef std::mt19937 random_generator_t;

        typedef std::array<
                random_generator_t::result_type,
                random_generator_t::state_size> random_state_t;

    private: // =============================================== MEMBER VARIABLES

        const std::size_t num_vars;
        OrderConditionEvaluator<T> evaluator;
        const std::function<T(const T *)> stdfun;
        const std::function<void(T *, const T *)> stdgrad;
        dznl::BFGSOptimizer<T> optimizer;
        std::uint64_t uuid[5];
        random_generator_t prng;

    private: // ===================================== CONSTRUCTOR HELPER METHODS

        static random_generator_t properly_seeded_random_generator() {
            random_state_t seed_data;
            std::random_device nondet_random_source;
            std::generate(seed_data.begin(), seed_data.end(),
                          std::ref(nondet_random_source));
            seed_data[0] = static_cast<random_generator_t::result_type>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch()
                    ).count() % std::numeric_limits<
                            random_generator_t::result_type>::max());
            std::seed_seq seed_sequence(seed_data.begin(), seed_data.end());
            random_generator_t random_generator(seed_sequence);
            return random_generator;
        }

        void randomize_uuid() {
            std::uniform_int_distribution<std::uint64_t> unif;
            uuid[0] = unif(prng) & 0xFFFFFFFF;
            uuid[1] = unif(prng) & 0xFFFF;
            uuid[2] = unif(prng) & 0xFFFF;
            uuid[3] = unif(prng) & 0xFFFF;
            uuid[4] = unif(prng) & 0xFFFFFFFFFFFF;
        }

    public: // ===================================================== CONSTRUCTOR

        OptimizerDriver(int order, std::size_t num_stages)
            : num_vars(num_stages * (num_stages + 1) / 2),
              evaluator(order, num_stages),
              stdfun([&](const T *x) {
                  return evaluator.objective_function(x);
              }),
              stdgrad([&](T *g, const T *x) {
                  evaluator.objective_gradient(g, x);
              }),
              optimizer(num_vars, stdfun, stdgrad),
              prng(properly_seeded_random_generator()) {
            randomize_uuid();
        }

    public: // ==================================================== INITIALIZERS

        void initialize_random() {
            std::uniform_real_distribution<T> unif(T(0), T(1));
            VectorXT x(num_vars);
            for (std::size_t i = 0; i < num_vars; ++i) {
                x[i] = unif(prng);
            }
            optimizer.set_current_point(x);
            optimizer.set_iteration_count(0);
            randomize_uuid();
        }

        void initialize_from_file(const std::string &filename) {
            std::cout << "Opening input file \""
                      << filename << "\"..." << std::endl;
            std::ifstream input_file(filename);
            if (!input_file.good()) {
                std::cout << "ERROR: could not open input file." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            std::cout << "Successfully opened input file. Reading..."
                      << std::endl;
            VectorXT x(num_vars);
            for (std::size_t i = 0; i < num_vars; ++i) {
                input_file >> x[i];
                if (input_file.fail()) {
                    std::cout << "ERROR: Could not read input file entry "
                                 "at index " << i << "." << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
            std::cout << "Successfully read input file." << std::endl;
            optimizer.set_current_point(x);
            if (is_rktk_filename(filename)) {
                optimizer.set_iteration_count(
                        dec_substr_to_uint(filename, 52, 64));
                uuid[0] = hex_substr_to_uint(filename, 15, 23);
                uuid[1] = hex_substr_to_uint(filename, 24, 28);
                uuid[2] = hex_substr_to_uint(filename, 29, 33);
                uuid[3] = hex_substr_to_uint(filename, 34, 38);
                uuid[4] = hex_substr_to_uint(filename, 39, 51);
            } else {
                optimizer.set_iteration_count(0);
                randomize_uuid();
            }
        }

    public: // ======================================================= ACCESSORS

        std::size_t get_iteration_count() {
            return optimizer.get_iteration_count();
        }

    public: // ======================================================== MUTATORS

        void set_step_size(const T &h) { optimizer.set_step_size(h); }

    public: // ============================================ OPTIMIZATION METHODS

        bool step() {
            bool result = optimizer.step();
            if (!result) {
                std::cout << "NOTICE: Optimal step size reduced to zero. "
                          << "BFGS iteration has converged "
                          << "to the requested precision." << std::endl;
            }
            return result;
        }

    public: // ========================================== CONSOLE OUTPUT METHODS

        void print(int print_precision) {
            std::ios_base::fmtflags old_flags = std::cout.flags();
            std::cout << std::setfill('0') << std::setw(12)
                      << optimizer.get_iteration_count();
            std::cout << std::showpos << std::scientific;
            std::cout << std::setprecision(
                    print_precision > 0 ? print_precision
                                        : std::numeric_limits<T>::max_digits10);
            std::cout << " | " << optimizer.get_current_objective_value()
                      << " | " << optimizer.get_current_gradient().norm()
                      << " | " << optimizer.get_last_step_size()
                      << " | " << optimizer.get_current_point().norm()
                      << " | ";
            switch (optimizer.get_last_step_type()) {
                case dznl::StepType::NONE:
                    std::cout << "NONE";
                    break;
                case dznl::StepType::GRAD:
                    std::cout << "GRAD";
                    break;
                case dznl::StepType::BFGS:
                    std::cout << "BFGS";
                    break;
            }
            std::cout << std::endl;
            std::cout.flags(old_flags);
        }

    public: // ============================================= FILE OUTPUT METHODS

        std::string current_filename() {
            using std::log10;
            auto f_score = static_cast<int>(T(-100) * log10(
                    optimizer.get_current_objective_value()));
            if (f_score < 0) { f_score = 0; }
            if (f_score > 9999) { f_score = 9999; }
            auto g_score = static_cast<int>(T(-100) * log10(
                    optimizer.get_current_gradient().norm()));
            if (g_score < 0) { g_score = 0; }
            if (g_score > 9999) { g_score = 9999; }
            std::ostringstream filename;
            filename << std::setfill('0') << std::dec;
            filename << std::setw(4) << f_score << '-';
            filename << std::setw(4) << g_score << "-RKTK-";
            filename << std::hex << std::uppercase;
            filename << std::setw(8) << uuid[0] << '-';
            filename << std::setw(4) << uuid[1] << '-';
            filename << std::setw(4) << uuid[2] << '-';
            filename << std::setw(4) << uuid[3] << '-';
            filename << std::setw(12) << uuid[4] << '-';
            filename << std::dec;
            filename << std::setw(12)
                     << optimizer.get_iteration_count() << ".txt";
            return filename.str();
        }

        void write_to_file(std::string filename = "") {
            if (filename.empty()) { filename = current_filename(); }
            std::ofstream output_file(filename);
            output_file << std::showpos << std::scientific << std::setprecision(
                    std::numeric_limits<T>::max_digits10);
            const VectorXT &x = optimizer.get_current_point();
            for (std::size_t i = 0; i < num_vars; ++i) {
                output_file << x[i] << '\n';
            }
            output_file << '\n';
            output_file << "Objective function value: "
                        << optimizer.get_current_objective_value() << '\n';
            output_file << "Objective gradient norm:  "
                        << optimizer.get_current_gradient().norm() << '\n';
            output_file << "Most recent step size:    "
                        << optimizer.get_last_step_size() << '\n';
            output_file << "Distance from origin:     "
                        << optimizer.get_current_point().norm() << '\n';
        }

    private: //========================== STATIC FILENAME PARSING HELPER METHODS

        static bool is_dec_digit(char c) {
            return ('0' <= c) && (c <= '9');
        }

        static bool is_hex_digit(char c) {
            return (('0' <= c) && (c <= '9'))
                   || (('a' <= c) && (c <= 'f'))
                   || (('A' <= c) && (c <= 'F'));
        }

        static bool is_dec_substr(const std::string &str,
                                  std::string::size_type begin,
                                  std::string::size_type end) {
            for (auto i = begin; i < end; ++i) {
                if (!is_dec_digit(str[i])) { return false; }
            }
            return true;
        }

        static bool is_hex_substr(const std::string &str,
                                  std::string::size_type begin,
                                  std::string::size_type end) {
            for (auto i = begin; i < end; ++i) {
                if (!is_hex_digit(str[i])) { return false; }
            }
            return true;
        }

    private: // ================================ STATIC FILENAME PARSING METHODS

        static bool is_rktk_filename(const std::string &filename) {
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

        static std::uint64_t dec_substr_to_uint(const std::string &str,
                                                std::string::size_type begin,
                                                std::string::size_type end) {
            std::istringstream substr_stream(str.substr(begin, end - begin));
            std::uint64_t result;
            substr_stream >> std::dec >> result;
            return result;
        }

        static std::uint64_t hex_substr_to_uint(const std::string &str,
                                                std::string::size_type begin,
                                                std::string::size_type end) {
            std::istringstream substr_stream(str.substr(begin, end - begin));
            std::uint64_t result;
            substr_stream >> std::hex >> result;
            return result;
        }

    }; // class OptimizerDriver

} // namespace rktk

#endif // RKTK_OPTIMIZER_DRIVER_HPP_INCLUDED
