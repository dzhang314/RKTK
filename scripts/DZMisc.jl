__precompile__(false)
module DZMisc

export dbl, rooted_tree_count, orthonormalize_columns!,
    approx_norm, approx_norm2, approx_normalize!

dbl(x::T) where {T <: Number} = x + x

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

end # module DZMisc
