__precompile__(false)
baremodule QuadFloat

export Quad, QuadF16, QuadF32, QuadF64, scale, exact

import Base: +, -, *, /, inv, sqrt, <, <=, zero, one, iszero, isone, signbit,
    Float16, Float32, Float64
import DZMisc: dbl

using Base: Number, IEEEFloat, AbstractFloat, BigInt, BigFloat, Rational,
    precision, setprecision, @inline, ==, &, |
using IEEEAccurateArithmetic

struct Quad{T <: IEEEFloat} <: AbstractFloat
    x0::T
    x1::T
    x2::T
    x3::T
end

const QuadF16 = Quad{Float16}
const QuadF32 = Quad{Float32}
const QuadF64 = Quad{Float64}

Float16(x::QuadF16) = x.x0
Float32(x::QuadF32) = x.x0
Float64(x::QuadF64) = x.x0

Float16(x::QuadF32) = Float16(x.x0)
Float16(x::QuadF64) = Float16(x.x0)
Float32(x::QuadF64) = Float32(x.x0)

Float32(x::QuadF16) = Float32(x.x0) + Float32(x.x1) + Float32(x.x2) + Float32(x.x3)
Float64(x::QuadF16) = Float64(x.x0) + Float64(x.x1) + Float64(x.x2) + Float64(x.x3)
Float64(x::QuadF32) = Float64(x.x0) + Float64(x.x1) + Float64(x.x2) + Float64(x.x3)

Quad{T}(b::Bool) where {T <: IEEEFloat} =
    Quad{T}(T(b), zero(T), zero(T), zero(T))

function Quad{F}(x::T) where {F <: IEEEFloat, T <: Integer}
    x0 = F(x)
    r1 = x - T(x0)
    x1 = F(r1)
    r2 = r1 - T(x1)
    x2 = F(r2)
    r3 = r2 - T(x2)
    x3 = F(r3)
    Quad{F}(x0, x1, x2, x3)
end

function Quad{F}(x::BigFloat) where {F <: IEEEFloat}
    setprecision(Int(precision(x))) do
        x0 = F(x)
        r1 = x - BigFloat(x0)
        x1 = F(r1)
        r2 = r1 - BigFloat(x1)
        x2 = F(r2)
        r3 = r2 - BigFloat(x2)
        x3 = F(r3)
        Quad{F}(x0, x1, x2, x3)
    end
end

Quad(x::T) where {T <: Integer} = Quad{Float64}(x)
Quad(x::BigFloat) = Quad{Float64}(x)
Quad(x::T) where {T <: IEEEFloat} = Quad{T}(x, zero(T), zero(T), zero(T))
Quad{T}(x::T) where {T <: IEEEFloat} = Quad{T}(x, zero(T), zero(T), zero(T))

zero(::Type{Quad{T}}) where {T <: IEEEFloat} =
    Quad{T}(zero(T), zero(T), zero(T), zero(T))
one(::Type{Quad{T}}) where {T <: IEEEFloat} =
    Quad{T}(one(T), zero(T), zero(T), zero(T))

zero(::Type{Quad}) = zero(Quad{Float64})
one(::Type{Quad}) = one(Quad{Float64})

iszero(x::Quad{T}) where {T <: IEEEFloat} =
    iszero(x.x0) & iszero(x.x1) & iszero(x.x2) & iszero(x.x3)
isone(x::Quad{T}) where {T <: IEEEFloat} =
    isone(x.x0) & iszero(x.x1) & iszero(x.x2) & iszero(x.x3)

signbit(x::Quad{T}) where {T <: IEEEFloat} = signbit(x.x0)

<(a::Quad{T}, b::Quad{T}) where {T <: AbstractFloat} =
    ((a.x0 < b.x0) |
        ((a.x0 == b.x0) & ((a.x1 < b.x1) |
            ((a.x1 == b.x1) & ((a.x2 < b.x2) |
                ((a.x2 == b.x2) & (a.x3 < b.x3)))))))

<=(a::Quad{T}, b::Quad{T}) where {T <: AbstractFloat} =
    ((a.x0 < b.x0) |
        ((a.x0 == b.x0) & ((a.x1 < b.x1) |
            ((a.x1 == b.x1) & ((a.x2 < b.x2) |
                ((a.x2 == b.x2) & (a.x3 <= b.x3)))))))

scale(a::T, x::Quad{T}) where {T <: IEEEFloat} =
    Quad{T}(a * x.x0, a * x.x1, a * x.x2, a * x.x3)

dbl(x::Quad{T}) where {T <: IEEEFloat} = scale(T(2), x)

@inline function +(a::Quad{T}, b::Quad{T}) where {T <: IEEEFloat}
    s0, t0 = two_sum(a.x0, b.x0)
    s1, t1 = two_sum(a.x1, b.x1)
    s2, t2 = two_sum(a.x2, b.x2)
    s3, t3 = two_sum(a.x3, b.x3)
    s1, t0 = two_sum(s1, t0)
    s2, t0, t1 = three_sum(s2, t0, t1)
    s3, t0 = three_sum2(s3, t0, t2)
    t0 = t0 + t1 + t3
    Quad{T}(renormalize(s0, s1, s2, s3, t0)...)
end

@inline function +(a::Quad{T}, b::T) where {T <: IEEEFloat}
    s0, t0 = two_sum(a.x0, b)
    s1, t0 = two_sum(a.x1, t0)
    s2, t0 = two_sum(a.x2, t0)
    s3, t0 = two_sum(a.x3, t0)
    Quad{T}(renormalize(s0, s1, s2, s3, t0)...)
end

@inline +(a::T, b::Quad{T}) where {T <: IEEEFloat} = b + a

@inline function -(a::Quad{T}, b::Quad{T}) where {T <: IEEEFloat}
    s0, t0 = two_diff(a.x0, b.x0)
    s1, t1 = two_diff(a.x1, b.x1)
    s2, t2 = two_diff(a.x2, b.x2)
    s3, t3 = two_diff(a.x3, b.x3)
    s1, t0 = two_sum(s1, t0)
    s2, t0, t1 = three_sum(s2, t0, t1)
    s3, t0 = three_sum2(s3, t0, t2)
    t0 = t0 + t1 + t3
    Quad{T}(renormalize(s0, s1, s2, s3, t0)...)
end

@inline function -(a::T, b::Quad{T}) where {T <: IEEEFloat}
    s0, t0 = two_diff(a, b.x0)
    s1, t0 = two_diff(t0, b.x1)
    s2, t0 = two_diff(t0, b.x2)
    s3, t0 = two_diff(t0, b.x3)
    Quad{T}(renormalize(s0, s1, s2, s3, t0)...)
end

@inline function -(a::Quad{T}) where {T <: IEEEFloat}
    Quad{T}(-a.x0, -a.x1, -a.x2, -a.x3)
end

@inline function *(a::Quad{T}, b::Quad{T}) where {T <: IEEEFloat}
    p0, q0 = two_prod(a.x0, b.x0)
    p1, q1 = two_prod(a.x0, b.x1)
    p2, q2 = two_prod(a.x1, b.x0)
    p3, q3 = two_prod(a.x0, b.x2)
    p4, q4 = two_prod(a.x1, b.x1)
    p5, q5 = two_prod(a.x2, b.x0)
    p1, p2, q0 = three_sum(p1, p2, q0)
    p2, q1, q2 = three_sum(p2, q1, q2)
    p3, p4, p5 = three_sum(p3, p4, p5)
    s0, t0 = two_sum(p2, p3)
    s1, t1 = two_sum(q1, p4)
    s2 = q2 + p5
    s1, t0 = two_sum(s1, t0)
    s2 += t0 + t1
    s1 += a.x0*b.x3 + a.x1*b.x2 + a.x2*b.x1 + a.x3*b.x0 + q0 + q3 + q4 + q5
    Quad{T}(renormalize(p0, p1, s0, s1, s2)...)
end

@inline function *(a::Quad{T}, b::T) where {T <: IEEEFloat}
    p0, q0 = two_prod(a.x0, b)
    p2, q2 = two_prod(a.x1, b)
    p5, q5 = two_prod(a.x2, b)
    p1, p2 = two_sum(p2, q0)
    p2, q1 = two_sum(p2, q2)
    s0, t0 = two_sum(p2, p5)
    s1, s2 = two_sum(q1, t0)
    s1 += a.x3*b + q5
    Quad{T}(renormalize(p0, p1, s0, s1, s2)...)
end

@inline *(a::T, b::Quad{T}) where {T <: IEEEFloat} = b * a

@inline function /(a::Quad{T}, b::Quad{T}) where {T <: IEEEFloat}
    q0 = a.x0 / b.x0
    r = a - q0 * b
    q1 = r.x0 / b.x0
    r -= q1 * b
    q2 = r.x0 / b.x0
    r -= q2 * b
    q3 = r.x0 / b.x0
    Quad{T}(renormalize(q0, q1, q2, q3)...)
end

@inline function inv(b::Quad{T}) where {T <: IEEEFloat}
    q0 = inv(b.x0)
    r = one(T) - q0 * b
    q1 = r.x0 / b.x0
    r -= q1 * b
    q2 = r.x0 / b.x0
    r -= q2 * b
    q3 = r.x0 / b.x0
    Quad{T}(renormalize(q0, q1, q2, q3)...)
end

function sqrt(x::Quad{T}) where {T <: IEEEFloat}
    if iszero(x)
        x
    else
        r = Quad{T}(inv(sqrt(x.x0)))
        h = scale(T(0.5), x)
        r += r * (T(0.5) - h * r * r)
        r += r * (T(0.5) - h * r * r)
        r * x
    end
end

function exact(x::Quad{T}) where {T <: IEEEFloat}
    (Rational{BigInt}(x.x0) + Rational{BigInt}(x.x1) +
        Rational{BigInt}(x.x2) + Rational{BigInt}(x.x3))
end

end # baremodule QuadFloat
