__precompile__(false)
baremodule IEEEAccurateArithmetic

export two_sum, quick_two_sum, three_sum, three_sum2,
    two_diff, two_prod, two_sqr, renormalize

using Base: +, -, *, fma, IEEEFloat, @inline

@inline function two_sum(a::T, b::T) where {T <: IEEEFloat}
    s = a + b
    v = s - a
    e = (a - (s - v)) + (b - v)
    s, e
end

@inline function quick_two_sum(a::T, b::T) where {T <: IEEEFloat}
    s = a + b
    e = b - (s - a)
    s, e
end

@inline function three_sum(a::T, b::T, c::T) where {T <: IEEEFloat}
    t0, t1 = two_sum(a, b)
    r0, t2 = two_sum(c, t0)
    r1, r2 = two_sum(t1, t2)
    r0, r1, r2
end

@inline function three_sum2(a::T, b::T, c::T) where {T <: IEEEFloat}
    t0, t1 = two_sum(a, b)
    r0, t2 = two_sum(c, t0)
    r1 = t1 + t2
    r0, r1
end

@inline function two_diff(a::T, b::T) where {T <: IEEEFloat}
    d = a - b
    v = a - d
    e = (a - (d + v)) - (b - v)
    d, e
end

@inline function two_prod(a::T, b::T) where {T <: IEEEFloat}
    p = a * b
    e = fma(a, b, -p)
    p, e
end

@inline function two_sqr(a::T) where {T <: IEEEFloat}
    p = a * a
    e = fma(a, a, -p)
    p, e
end

@inline function renormalize(
        c0::T, c1::T, c2::T, c3::T, c4::T) where {T <: IEEEFloat}
    s, t3 = quick_two_sum(c3, c4)
    s, t2 = quick_two_sum(c2, s)
    s, t1 = quick_two_sum(c1, s)
    r0, t0 = quick_two_sum(c0, s)
    s, t2 = quick_two_sum(t2, t3)
    s, t1 = quick_two_sum(t1, s)
    r1, t0 = quick_two_sum(t0, s)
    s, t1 = quick_two_sum(t1, t2)
    r2, t0 = quick_two_sum(t0, s)
    r3 = t0 + t1
    r0, r1, r2, r3
end

@inline function renormalize(
        c0::T, c1::T, c2::T, c3::T) where {T <: IEEEFloat}
    s, t2 = quick_two_sum(c2, c3)
    s, t1 = quick_two_sum(c1, s)
    r0, t0 = quick_two_sum(c0, s)
    s, t1 = quick_two_sum(t1, t2)
    r1, t0 = quick_two_sum(t0, s)
    r2, r3 = quick_two_sum(t0, t1)
    r0, r1, r2, r3
end

end # baremodule IEEEAccurateArithmetic
