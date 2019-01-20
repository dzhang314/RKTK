module DZMisc

export dbl, scale, rooted_tree_count, orthonormalize_columns!,
    norm, norm2, normalize!, approx_norm, approx_norm2, approx_normalize!,
    quadratic_line_search

using LinearAlgebra: dot, mul!

################################################################################

dbl(x::T) where {T <: Number} = x + x

scale(a::Float64, x::T) where {T <: Number} = T(a * x)

################################################################################

function rooted_tree_count(n::Int)
    if n <= 0
        error("rooted_tree_count requires positive argument")
    end
    counts = zeros(Rational{BigInt}, n)
    counts[1] = 1
    for i = 2 : n, k = 1 : i - 1, m = 1 : div(i - 1, k)
        @inbounds counts[i] += k * counts[k] * counts[i - k * m] // (i - 1)
    end
    @assert all(isone(denominator(c)) for c in counts)
    sum(numerator.(counts))
end

################################################################################

function orthonormalize_columns!(mat::Matrix{T}) where {T <: Real}
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

function norm2(x::Vector{T}) where {T <: Number}
    result = zero(float(real(T)))
    @simd for i = 1 : length(x)
        @inbounds result += abs2(x[i])
    end
    result
end

norm(x::Vector{T}) where {T <: Number} = sqrt(norm2(x))

function normalize!(x::Vector{T}) where {T <: Number}
    a = inv(norm(x))
    @simd ivdep for i = 1 : length(x)
        @inbounds x[i] *= a
    end
end

################################################################################

function approx_norm2(x::Vector{T}) where {T <: Number}
    result = zero(Float64)
    @simd for i = 1 : length(x)
        @inbounds result += abs2(Float64(x[i]))
    end
    result
end

approx_norm(x::Vector{T}) where {T <: Number} = sqrt(approx_norm2(x))

function approx_normalize!(x::Vector{T}) where {T <: Number}
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
        s::Vector{T}, y::Vector{T}, t::Vector{T}) where {T <: Real}
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
