module DZPolynomials

export ExponentVector, Polynomial, reduce!, dict_add!, dict_sub!

################################################################################

struct ExponentVector
    data::Vector{Int}
end

const EMPTY_EXPONENT_VECTOR = ExponentVector([])

################################################################################

import Base: length, isempty, getindex, ==, hash, isless, show

@inline length(v::ExponentVector)                    = length(v.data)
@inline isempty(v::ExponentVector)                   = isempty(v.data)
@inline getindex(v::ExponentVector, i::Int)          = getindex(v.data, i)
@inline ==(v::ExponentVector, w::ExponentVector)     = (v.data == w.data)
@inline hash(v::ExponentVector, h::UInt)             = hash(v.data, h)
@inline isless(v::ExponentVector, w::ExponentVector) = isless(v.data, w.data)

const VARIABLE_NAMES       = Ref{Vector{String}}(String[])
const UNICODE_SUBSCRIPTS   = Dict(map(Pair, "0123456789", "₀₁₂₃₄₅₆₇₈₉"))
const UNICODE_SUPERSCRIPTS = Dict(map(Pair, "0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
subscript_string(n::Int)   = map(c -> UNICODE_SUBSCRIPTS[c]  , string(n))
superscript_string(n::Int) = map(c -> UNICODE_SUPERSCRIPTS[c], string(n))

function show(io::IO, v::ExponentVector)
    for (i, e) in enumerate(v.data)
        if i <= length(VARIABLE_NAMES[])
            var = VARIABLE_NAMES[][i]
        else
            var = "x" * subscript_string(i)
        end
        if e == 1
            write(io, var)
        elseif e > 1
            write(io, var, superscript_string(e))
        end
    end
end

################################################################################

import Base: +, -, lcm

function +(v::ExponentVector, w::ExponentVector)
    len_v = length(v)
    len_w = length(w)
    m, n = minmax(len_v, len_w)
    r = Vector{Int}(undef, n)
    @simd ivdep for i = 1 : m
        @inbounds r[i] = v[i] + w[i]
    end
    @simd ivdep for i = m + 1 : len_v
        @inbounds r[i] = v[i]
    end
    @simd ivdep for i = m + 1 : len_w
        @inbounds r[i] = w[i]
    end
    ExponentVector(r)
end

function -(v::ExponentVector, w::ExponentVector)
    len_v = length(v)
    len_w = length(w)
    if len_v < len_w
        error("invalid ExponentVector subtraction")
    elseif len_v == len_w
        n = 0
        for i = len_v : -1 : 1
            if v[i] == w[i]
                n += 1
            else
                break
            end
        end
        r = Vector{Int}(undef, len_v - n)
        @simd ivdep for i = 1 : len_v - n
            @inbounds r[i] = v[i] - w[i]
        end
        ExponentVector(r)
    else
        r = Vector{Int}(undef, len_v)
        @simd ivdep for i = 1 : len_w
            @inbounds r[i] = v[i] - w[i]
        end
        @simd ivdep for i = len_w + 1 : len_v
            @inbounds r[i] = v[i]
        end
        ExponentVector(r)
    end
end

function lcm(v::ExponentVector, w::ExponentVector)
    len_v = length(v)
    len_w = length(w)
    m, n = minmax(len_v, len_w)
    r = Vector{Int}(undef, n)
    @simd ivdep for i = 1 : m
        @inbounds r[i] = max(v[i], w[i])
    end
    @simd ivdep for i = m + 1 : len_v
        @inbounds r[i] = v[i]
    end
    @simd ivdep for i = m + 1 : len_w
        @inbounds r[i] = w[i]
    end
    ExponentVector(r)
end

function isdivisible(v::ExponentVector, w::ExponentVector)
    len_v = length(v)
    len_w = length(w)
    if len_v > len_w
        false
    else
        all(v[i] <= w[i] for i = 1 : len_v)
    end
end

################################################################################

struct Polynomial{R <: Real}
    data::Dict{ExponentVector,R}
end

################################################################################

import Base: zero, iszero, convert, show

@inline zero(::Type{Polynomial{R}}) where {R <: Real} =
    Polynomial{R}(Dict{ExponentVector,R}())

@inline iszero(p::Polynomial{R}) where {R <: Real} = iszero(length(p.data))

convert(::Type{Polynomial{R}}, x::Real) where {R <: Real} = Polynomial{R}(
    Dict{ExponentVector,R}(EMPTY_EXPONENT_VECTOR => convert(R, x)))

function show(io::IO, p::Polynomial{R}) where {R <: Real}
    data = [(k, v) for (k, v) in p.data]
    sort!(data, rev=true)
    if isempty(data)
        write(io, '0')
    end
    for (i, (k, v)) in enumerate(data)
        if (i == 1) && signbit(v)
            write(io, '-')
        elseif i > 1
            if signbit(v)
                write(io, " - ")
            else
                write(io, " + ")
            end
        end
        v = abs(v)
        if isempty(k) || !isone(v)
            # if isone(denominator(v))
                show(io, v)
            # else
            #     show(io, numerator(v))
            #     write(io, '/')
            #     show(io, denominator(v))
            #     if !isempty(k)
            #         #write(io, ' ')
            #     end
            # end
        end
        show(io, k)
    end
end

################################################################################

import Base: +, -, *, ^, %

function dict_add!(dict::Dict{K,V}, key::K, val::W) where {K, V, W}
    if haskey(dict, key)
        prev = dict[key]
        new = prev + val
        if iszero(new)
            delete!(dict, key)
        else
            dict[key] = new
        end
    else
        dict[key] = val
    end
end

function dict_sub!(dict::Dict{K,V}, key::K, val::W) where {K, V, W}
    if haskey(dict, key)
        prev = dict[key]
        new = prev - val
        if iszero(new)
            delete!(dict, key)
        else
            dict[key] = new
        end
    else
        dict[key] = -val
    end
end

function +(p::Polynomial{R}) where {R <: Real}
    r = Dict{ExponentVector,R}()
    for (k, v) in p.data
        r[k] = +v
    end
    Polynomial{R}(r)
end

function -(p::Polynomial{R}) where {R <: Real}
    r = Dict{ExponentVector,R}()
    for (k, v) in p.data
        r[k] = -v
    end
    Polynomial{R}(r)
end

function +(p::Polynomial{R}, q::Polynomial{R}) where {R <: Real}
    r = Dict{ExponentVector,R}()
    for (k, v) in p.data
        dict_add!(r, k, v)
    end
    for (k, v) in q.data
        dict_add!(r, k, v)
    end
    Polynomial{R}(r)
end

function +(p::Polynomial{R}, c::T) where {R <: Real, T <: Real}
    r = Dict{ExponentVector,R}()
    if !iszero(c)
        r[EMPTY_EXPONENT_VECTOR] = convert(R, c)
    end
    for (k, v) in p.data
        dict_add!(r, k, v)
    end
    Polynomial{R}(r)
end

@inline +(c::T, p::Polynomial{R}) where {R <: Real, T <: Real} = p + c

function -(p::Polynomial{R}, q::Polynomial{R}) where {R <: Real}
    r = Dict{ExponentVector,R}()
    for (k, v) in p.data
        dict_add!(r, k, v)
    end
    for (k, v) in q.data
        dict_sub!(r, k, v)
    end
    Polynomial{R}(r)
end

function -(p::Polynomial{R}, c::T) where {R <: Real, T <: Real}
    r = Dict{ExponentVector,R}()
    if !iszero(c)
        r[EMPTY_EXPONENT_VECTOR] = convert(R, -c)
    end
    for (k, v) in p.data
        dict_add!(r, k, v)
    end
    Polynomial{R}(r)
end

function -(c::T, p::Polynomial{R}) where {R <: Real, T <: Real}
    r = Dict{ExponentVector,R}()
    if !iszero(c)
        r[EMPTY_EXPONENT_VECTOR] = convert(R, c)
    end
    for (k, v) in p.data
        dict_sub!(r, k, v)
    end
    Polynomial{R}(r)
end

function *(p::Polynomial{R}, q::Polynomial{R}) where {R <: Real}
    r = Dict{ExponentVector,R}()
    for (kp, vp) in p.data
        for (kq, vq) in q.data
            dict_add!(r, kp + kq, vp * vq)
        end
    end
    Polynomial{R}(r)
end

function *(p::Polynomial{R}, c::T) where {R <: Real, T <: Real}
    r = Dict{ExponentVector,R}()
    if !iszero(c)
        for (kp, vp) in p.data
            r[kp] = vp * c
        end
    end
    Polynomial{R}(r)
end

@inline *(c::T, p::Polynomial{R}) where {R <: Real, T <: Real} = p * c

function ^(p::Polynomial{R}, n::Int) where {R <: Real}
    if n < 0
        error("cannot raise Polynomial to negative power")
    elseif n == 0
        convert(Polynomial{R}, 1)
    elseif n == 1
        p
    elseif n % 2 == 0
        q = p^div(n, 2)
        q * q
    else
        q = p^div(n, 2)
        q * q * p
    end
end

function reduce!(p::Polynomial{R}, g::Vector{Polynomial{R}}) where {R <: Real}
    result = Tuple{ExponentVector,R}[]
    ltg = [maximum(h.data) for h in g]
    while !iszero(p)
        moved = false
        ltp, lcp = maximum(p.data)
        for (h, (lth, lch)) in zip(g, ltg)
            if isdivisible(lth, ltp)
                t = ltp - lth
                c = lcp / lch
                for (k, v) in h.data
                    dict_sub!(p.data, k + t, v * c)
                end
                moved = true
                break
            end
        end
        if !moved
            push!(result, (ltp, lcp))
            delete!(p.data, ltp)
        end
    end
    for (k, v) in result
        p.data[k] = v
    end
end

end # module DZPolynomials
