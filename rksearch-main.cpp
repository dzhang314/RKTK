// C++ standard library headers
#include <cmath> // for std::isfinite
#include <cstdlib> // for std::exit
#include <cstring> // for std::strlen
#include <ctime> // for std::clock
#include <iostream> // for std::cout
#include <limits> // for std::numeric_limits

// Boost library headers
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/eigen.hpp>

// RKTK headers
#include "OptimizerDriver.hpp"

template <unsigned Digits>
using boost_float = boost::multiprecision::number<
    boost::multiprecision::cpp_bin_float<Digits,
        boost::multiprecision::digit_base_2>,
    boost::multiprecision::et_off>;

long long int get_positive_integer_argument(char *str) {
    char *end;
    const long long int result = std::strtoll(str, &end, 10);
    const bool read_whole_arg = (
            std::strlen(str) == static_cast<std::size_t>(end - str));
    const int is_positive = (result > 0);
    if (!read_whole_arg || !is_positive) {
        std::cerr << "ERROR: Expected command-line argument '" << str
                  << "' to be a positive integer." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return result;
}

#define RETURN_PRECISION(T)                                                    \
        return std::numeric_limits<T>::digits;                                 \
    } else if (precision == std::numeric_limits<T>::digits) {                  \
        return std::numeric_limits<T>::digits;                                 \
    }

#define HANDLE_SMALLEST_PRECISION(T)                                           \
    if (precision < std::numeric_limits<T>::digits) {                          \
        std::cout << "Rounding up to " << std::numeric_limits<T>::digits       \
                  << "-bit precision, which is the smallest "                  \
                        "available machine precision." << std::endl;           \
        RETURN_PRECISION(T)

#define HANDLE_MACHINE_PRECISION(T)                                            \
    else if (precision < std::numeric_limits<T>::digits) {                     \
        std::cout << "Rounding up to " << std::numeric_limits<T>::digits       \
                  << "-bit precision, which is the next "                      \
                        "available machine precision." << std::endl;           \
        RETURN_PRECISION(T)

#define HANDLE_EXTENDED_PRECISION(T)                                           \
    else if (precision < std::numeric_limits<T>::digits) {                     \
        std::cout << "Rounding up to " << std::numeric_limits<T>::digits       \
                  << "-bit precision, which is the next "                      \
                        "available extended precision." << std::endl;          \
        RETURN_PRECISION(T)

#define HANDLE_LARGEST_PRECISION(T)                                            \
    HANDLE_EXTENDED_PRECISION(T) else {                                        \
        std::cout << "WARNING: Requested precision exceeds "                   \
                     "available precision. Rounding down to "                  \
                  << std::numeric_limits<T>::digits                            \
                  << "-bit precision, which is the highest "                   \
                     "available extended precision." << std::endl;             \
        return std::numeric_limits<T>::digits;                                 \
    }

int get_precision(int argc, char **argv, int index) {
    if (argc > index) {
        char *end;
        const long long int precision = std::strtoll(argv[index], &end, 10);
        const bool read_whole_arg = (std::strlen(argv[index]) ==
                                     static_cast<std::size_t>(
                                             end - argv[index]));
        const bool is_positive = (precision > 0);
        if (read_whole_arg && is_positive) {
            std::cout << "Requested " << precision << "-bit precision."
                      << std::endl;
            HANDLE_SMALLEST_PRECISION(float)
            HANDLE_MACHINE_PRECISION(double)
            HANDLE_MACHINE_PRECISION(long double)
            HANDLE_EXTENDED_PRECISION(boost_float<128>)
            HANDLE_EXTENDED_PRECISION(boost_float<256>)
            HANDLE_EXTENDED_PRECISION(boost_float<384>)
            HANDLE_EXTENDED_PRECISION(boost_float<512>)
            HANDLE_EXTENDED_PRECISION(boost_float<640>)
            HANDLE_EXTENDED_PRECISION(boost_float<768>)
            HANDLE_EXTENDED_PRECISION(boost_float<896>)
            HANDLE_LARGEST_PRECISION(boost_float<1024>)
        } else {
            std::cout << "WARNING: Could not interpret command-line argument "
                      << argv[index] << " as a positive integer."
                                        " Defaulting to double ("
                      << std::numeric_limits<double>::digits
                      << "-bit) precision." << std::endl;
            return std::numeric_limits<double>::digits;
        }
    } else {
        std::cout << "Defaulting to double ("
                  << std::numeric_limits<double>::digits
                  << "-bit) precision." << std::endl;
        return std::numeric_limits<double>::digits;
    }
}

double get_print_period(int argc, char **argv, int index) {
    if (argc > index) {
        char *end;
        const double print_period = std::strtod(argv[index], &end);
        const bool read_whole_arg = (
                std::strlen(argv[index]) ==
                static_cast<std::size_t>(end - argv[index]));
        const bool is_finite = std::isfinite(print_period);
        const bool is_positive = (print_period >= 0.0);
        if (read_whole_arg && is_finite && is_positive) {
            return print_period;
        }
    }
    return 0.5;
}

int get_print_precision(int argc, char **argv, int index) {
    if (argc > index) {
        char *end;
        const long long print_precision = std::strtoll(argv[index], &end, 10);
        const bool read_whole_arg = (
                std::strlen(argv[index]) ==
                static_cast<std::size_t>(end - argv[index]));
        const bool is_non_negative = (print_precision >= 0);
        const bool in_range = (print_precision <=
                               std::numeric_limits<int>::max());
        if (read_whole_arg && is_non_negative && in_range) {
            return static_cast<int>(print_precision);
        }
    }
    return 0;
}

enum class SearchMode { EXPLORE, REFINE };

template <typename T>
void run_main_loop(int order, std::size_t num_steps,
                   char *filename, SearchMode mode, int print_prec,
                   std::clock_t clocks_between_prints) {
    rktk::OptimizerDriver<T> optimizer(order, num_steps);
    if (mode == SearchMode::REFINE) {
        optimizer.initialize_from_file(std::string(filename));
    }
    std::clock_t last_print_clock;
    reset_d:
    if (mode == SearchMode::EXPLORE) { optimizer.initialize_random(); }
    optimizer.print(print_prec);
    optimizer.write_to_file();
    last_print_clock = std::clock();
    optimizer.set_step_size(65536 * std::numeric_limits<T>::epsilon());
    while (true) {
        if (!optimizer.step()) {
            optimizer.print(print_prec);
            std::cout << "Located candidate local minimum." << std::endl;
            optimizer.write_to_file();
            if (mode == SearchMode::EXPLORE) { goto reset_d; }
            std::exit(EXIT_SUCCESS);
        }
        if (optimizer.get_iteration_count() % 1000 == 0) {
            optimizer.write_to_file();
        }
        if (mode == SearchMode::EXPLORE &&
            optimizer.get_iteration_count() >= 100000000) {
            std::cout << "NOTICE: Exceeded maximum number "
                         "of BFGS iterations. Restarting from "
                         "new random point." << std::endl;
            goto reset_d;
        }
        const std::clock_t current_clock = std::clock();
        if (current_clock - last_print_clock >= clocks_between_prints) {
            optimizer.print(print_prec);
            last_print_clock = current_clock;
        }
    }
}

#define RUN_MAIN_LOOP(T)                                                       \
    if (prec == std::numeric_limits<T>::digits) {                              \
        run_main_loop<T>(order, num_stages, argv[6], mode,                     \
                         print_prec, clocks_between_prints);                   \
    }

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " order num-stages num-bits "
                  << "print-period print-precision [input-filename]"
                  << std::endl;
        return EXIT_FAILURE;
    }
    const unsigned long long int order = static_cast<unsigned long long int>(
            get_positive_integer_argument(argv[1]));
    const std::size_t num_stages = static_cast<std::size_t>(
            get_positive_integer_argument(argv[2]));
    if (order > 15) {
        std::cerr << "ERROR: Runge-Kutta methods of order greater than 15 "
                  << "are not yet supported." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (order > num_stages) {
        std::cerr << "ERROR: The order of a Runge-Kutta method cannot exceed "
                  << "its number of stages." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    const int prec = get_precision(argc, argv, 3);
    const auto clocks_between_prints = static_cast<std::clock_t>(
            get_print_period(argc, argv, 4) * CLOCKS_PER_SEC);
    const int print_prec = get_print_precision(argc, argv, 5);
    const SearchMode mode = (argc > 6)
                            ? SearchMode::REFINE
                            : SearchMode::EXPLORE;
    RUN_MAIN_LOOP(float)
    else RUN_MAIN_LOOP(double)
    else RUN_MAIN_LOOP(long double)
    else RUN_MAIN_LOOP(boost_float<128>)
    else RUN_MAIN_LOOP(boost_float<256>)
    else RUN_MAIN_LOOP(boost_float<384>)
    else RUN_MAIN_LOOP(boost_float<512>)
    else RUN_MAIN_LOOP(boost_float<640>)
    else RUN_MAIN_LOOP(boost_float<768>)
    else RUN_MAIN_LOOP(boost_float<896>)
    else RUN_MAIN_LOOP(boost_float<1024>)
}
