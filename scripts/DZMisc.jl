__precompile__(false)
module DZMisc

export dbl, log, rooted_tree_count, orthonormalize_columns!

dbl(x::T) where {T <: Number} = x + x

function log(args...)
    println(args...)
    flush(stdout)
end

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

function orthonormalize_columns!(mat::Matrix{T}) where {T}
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

end # module DZMisc
