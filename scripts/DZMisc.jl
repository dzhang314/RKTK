module DZMisc

export dbl, scale, integer_partitions,
    RootedTree, rooted_trees, rooted_tree_count, butcher_density,
    orthonormalize_columns!,
    norm, norm2, normalize!, approx_norm, approx_norm2, approx_normalize!,
    quadratic_line_search

using LinearAlgebra: dot, mul!

################################################################################

@inline function dbl(x::T)::T where {T <: Number}
    x + x
end

@inline function scale(a::Float64, x::T)::T where {T <: Number}
    a * x
end

################################################################################

function _int_parts_impl(n::Int, max::Int)::Vector{Vector{Pair{Int,Int}}}
    if n == 0
        [Pair{Int,Int}[]]
    elseif max == 0
        Vector{Pair{Int,Int}}[]
    else
        result = Vector{Pair{Int,Int}}[]
        for m = max : -1 : 1, q = div(n, m) : -1 : 1
            for p in _int_parts_impl(n - q * m, m - 1)
                push!(result, push!(p, m => q))
            end
        end
        result
    end
end

function _int_parts_impl(n::Int, max::Int, len::Int)::Vector{Vector{Int}}
    if n == 0
        [zeros(Int, len)]
    elseif max * len < n
        Vector{Int}[]
    else
        result = Vector{Int}[]
        for m = min(n, max) : -1 : div(n + len - 1, len)
            for p in _int_parts_impl(n - m, m, len - 1)
                push!(result, push!(p, m))
            end
        end
        result
    end
end

integer_partitions(n::Int)::Vector{Vector{Pair{Int,Int}}} =
    reverse!.(_int_parts_impl(n, n))
integer_partitions(n::Int, len::Int)::Vector{Vector{Int}} =
    reverse!.(_int_parts_impl(n, n, len))

################################################################################

function next_permutation!(items::Vector{T})::Bool where {T}
    num_items = length(items)
    if num_items == 0; return false; end
    current_item = items[num_items]
    pivot_index = num_items - 1
    while pivot_index != 0
        next_item = items[pivot_index]
        if next_item >= current_item
            pivot_index -= 1
            current_item = next_item
        else; break; end
    end
    if pivot_index == 0; return false; end
    pivot = items[pivot_index]
    successor_index = num_items
    while items[successor_index] <= pivot; successor_index -= 1; end
    items[pivot_index], items[successor_index] =
        items[successor_index], items[pivot_index]
    reverse!(view(items, pivot_index + 1 : num_items))
    return true
end

function previous_permutation!(items::Vector{T})::Bool where {T}
    num_items = length(items)
    if num_items == 0; return false; end
    current_item = items[num_items]
    pivot_index = num_items - 1
    while pivot_index != 0
        next_item = items[pivot_index]
        if next_item <= current_item
            pivot_index -= 1
            current_item = next_item
        else; break; end
    end
    if pivot_index == 0; return false; end
    pivot = items[pivot_index]
    successor_index = num_items
    while items[successor_index] >= pivot; successor_index -= 1; end
    items[pivot_index], items[successor_index] =
        items[successor_index], items[pivot_index]
    reverse!(view(items, pivot_index + 1 : num_items))
    return true
end

function combinations_with_replacement(
        items::Vector{T}, n::Int)::Vector{Vector{Pair{T,Int}}} where {T}
        combinations = Vector{Pair{T,Int}}[]
    for p in integer_partitions(n, length(items))
        while true
            comb = Pair{T,Int}[]
            for (item, k) in zip(items, p)
                if k > 0; push!(comb, item => k); end
            end
            push!(combinations, comb)
            if !previous_permutation!(p); break; end
        end
    end
    combinations
end

################################################################################

struct RootedTree
    order::Int
    children::Vector{Pair{RootedTree,Int}}
end

# function Base.show(io::IO, tree::RootedTree)::Nothing
#     print(io, '[')
#     for (subtree, multiplicity) in tree.children
#         print(io, subtree)
#         if multiplicity != 1
#             print(io, '^', multiplicity)
#         end
#     end
#     print(io, ']')
# end

function rooted_trees(n::Int)::Vector{Vector{RootedTree}}
    result = Vector{RootedTree}[]
    for k = 1 : n
        trees = RootedTree[]
        for partition in integer_partitions(k - 1)
            combination_candidates = [
                combinations_with_replacement(result[order], multiplicity)
                for (order, multiplicity) in partition]
            for combination_sequence in Base.product(combination_candidates...)
                push!(trees, RootedTree(k, vcat(combination_sequence...)))
            end
        end
        push!(result, trees)
    end
    result
end

function rooted_tree_count(n::Int)::BigInt
    if n < 0; error("rooted_tree_count requires non-negative argument"); end
    if n == 0; return 0; end
    counts = zeros(Rational{BigInt}, n)
    counts[1] = 1
    for i = 2 : n, k = 1 : i - 1, m = 1 : div(i - 1, k)
        @inbounds counts[i] += k * counts[k] * counts[i - k * m] // (i - 1)
    end
    @assert all(isone(denominator(c)) for c in counts)
    sum(numerator.(counts))
end

function butcher_density(tree::RootedTree)::Int
    result = tree.order
    for (subtree, multiplicity) in tree.children
        result *= butcher_density(subtree)^multiplicity
    end
    result
end

################################################################################

function orthonormalize_columns!(mat::Matrix{T})::Nothing where {T <: Real}
    m = size(mat, 1)
    n = size(mat, 2)
    for j = 1 : n
        for k = 1 : j - 1
            acc = zero(T)
            @simd for i = 1 : m
                @inbounds acc += mat[i, j] * mat[i, k]
            end
            @simd ivdep for i = 1 : m
                @inbounds mat[i, j] -= acc * mat[i, k]
            end
        end
        acc = zero(T)
        @simd for i = 1 : m
            @inbounds acc += abs2(mat[i, j])
        end
        acc = inv(sqrt(acc))
        @simd ivdep for i = 1 : m
            @inbounds mat[i, j] *= acc
        end
    end
end

################################################################################

function norm2(x::Vector{T})::T where {T <: Number}
    result = zero(float(real(T)))
    @simd for i = 1 : length(x)
        @inbounds result += abs2(x[i])
    end
    result
end

@inline function norm(x::Vector{T})::T where {T <: Number}
    sqrt(norm2(x))
end

function normalize!(x::Vector{T})::Nothing where {T <: Number}
    a = inv(norm(x))
    @simd ivdep for i = 1 : length(x)
        @inbounds x[i] *= a
    end
end

################################################################################

function approx_norm2(x::Vector{T})::Float64 where {T <: Number}
    result = zero(Float64)
    @simd for i = 1 : length(x)
        @inbounds result += abs2(Float64(x[i]))
    end
    result
end

@inline function approx_norm(x::Vector{T})::Float64 where {T <: Number}
    sqrt(approx_norm2(x))
end

function approx_normalize!(x::Vector{T})::Nothing where {T <: Number}
    a = inv(approx_norm(x))
    @simd ivdep for i = 1 : length(x)
        @inbounds x[i] *= a
    end
end

################################################################################

function _qls_best(fb, x1, f1, x2, f2, x3, f3)
    xb = zero(x1)
    if f1 < fb
        xb, fb = x1, f1
    end
    if f2 < fb
        xb, fb = x2, f2
    end
    if f3 < fb
        xb, fb = x3, f3
    end
    xb, fb
end

function _qls_minimum_high(f0, f1, f2)
    q1 = f1 + f1
    q2 = q1 + q1
    q3 = f0 + f0
    q4 = f2 + f0
    q5 = q1 - q4
    (q2 - q3 - q4) / (q5 + q5)
end

function _qls_minimum_low(f0, f1, f2)
    q1 = f2 + f2
    q2 = q1 + q1
    q3 = f0 + f0
    q4 = f0 + f1
    q5 = q4 - q1
    q6 = q5 + q5
    (q4 + q3 - q2) / (q6 + q6)
end

function quadratic_line_search(f, f0, x1, args...)
    f1 = f(x1, args...)
    if isnan(f1)
        return zero(x1), f0
    end
    if f1 < f0
        while true
            x2 = scale(2.0, x1)
            f2 = f(x2, args...)
            if f2 >= f1
                x3 = x1 * _qls_minimum_high(f0, f1, f2)
                f3 = f(x3, args...)
                return _qls_best(f0, x1, f1, x2, f2, x3, f3)
            else
                x1, f1 = x2, f2
            end
        end
    else
        while true
            x2 = scale(0.5, x1)
            f2 = f(x2, args...)
            if isnan(f2)
                return zero(x1), f0
            end
            if f2 <= f0
                x3 = x1 * _qls_minimum_low(f0, f1, f2)
                f3 = f(x3, args...)
                return _qls_best(f0, x2, f2, x1, f1, x3, f3)
            else
                x1, f1 = x2, f2
            end
        end
    end
end

################################################################################

function update_inverse_hessian!(B_inv::Matrix{T}, h::T,
        s::Vector{T}, y::Vector{T}, t::Vector{T})::Nothing where {T <: Real}
    b = dot(s, y)
    s .*= inv(b)
    mul!(t, B_inv, y)
    a = h * b + dot(y, t)
    for j = 1 : size(B_inv, 2)
        sj = s[j]
        tj = t[j]
        @simd ivdep for i = 1 : size(B_inv, 1)
            @inbounds B_inv[i, j] += a * (s[i] * sj) - (t[i] * sj + s[i] * tj)
        end
    end
end

end # module DZMisc
