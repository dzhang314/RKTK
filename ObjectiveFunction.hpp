#ifndef RKTK_OBJECTIVE_FUNCTION_HPP_INCLUDED
#define RKTK_OBJECTIVE_FUNCTION_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

#define NUM_VARS 136

namespace rktk {

    template <typename T>
    T objective_function(const T *x);

    template <typename T>
    T objective_function_partial(const T *x, std::size_t i);

    template <typename T>
    void objective_gradient(T *__restrict__ g, const T *__restrict__ x);

} // namespace rktk

#endif // RKTK_OBJECTIVE_FUNCTION_HPP_INCLUDED
