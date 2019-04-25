module MultiprecisionFloats

export Float64x2, Float64x3, Float64x4,
    Float64x5, Float64x6, Float64x7, Float64x8

using MultiprecisionFloatCodeGenerator

@inline function two_sum(a::Float64, b::Float64)
    s = a + b
    v = s - a
    s, (a - (s - v)) + (b - v)
end

@inline function quick_two_sum(a::Float64, b::Float64)
    s = a + b
    s, b - (s - a)
end

@inline function two_prod(a::Float64, b::Float64)
    p = a * b
    p, fma(a, b, -p)
end

################################################################## DEFINITION OF
################################################################### MULTIFLOAT64

struct MultiFloat64{N} <: AbstractFloat
    x::NTuple{N,Float64}
end

##################################################### CONVERSION TO MULTIFLOAT64
########################################################### FROM PRIMITIVE TYPES

@inline MultiFloat64{N}(x::MultiFloat64{N}) where {N} = x

@inline MultiFloat64{N}(x::Float64) where {N} =
    MultiFloat64{N}((x, ntuple(_ -> 0.0, N - 1)...))

# Values of the types Bool, Int8, UInt8, Int16, UInt16, Float16, Int32, UInt32,
# and Float32 can be converted losslessly to a single Float64, which has 53
# bits of integer precision.

@inline MultiFloat64{N}(x::Bool   ) where {N} = MultiFloat64{N}(Float64(x))
@inline MultiFloat64{N}(x::Int8   ) where {N} = MultiFloat64{N}(Float64(x))
@inline MultiFloat64{N}(x::UInt8  ) where {N} = MultiFloat64{N}(Float64(x))
@inline MultiFloat64{N}(x::Int16  ) where {N} = MultiFloat64{N}(Float64(x))
@inline MultiFloat64{N}(x::UInt16 ) where {N} = MultiFloat64{N}(Float64(x))
@inline MultiFloat64{N}(x::Float16) where {N} = MultiFloat64{N}(Float64(x))
@inline MultiFloat64{N}(x::Int32  ) where {N} = MultiFloat64{N}(Float64(x))
@inline MultiFloat64{N}(x::UInt32 ) where {N} = MultiFloat64{N}(Float64(x))
@inline MultiFloat64{N}(x::Float32) where {N} = MultiFloat64{N}(Float64(x))

# Values of the types Int64, UInt64, Int128, and UInt128 cannot be converted
# losslessly to a single Float64 and must be split into multiple components.

@inline MultiFloat64{1}(x::Int64  ) = MultiFloat64{1}(Float64(x))
@inline MultiFloat64{1}(x::UInt64 ) = MultiFloat64{1}(Float64(x))
@inline MultiFloat64{1}(x::Int128 ) = MultiFloat64{1}(Float64(x))
@inline MultiFloat64{1}(x::UInt128) = MultiFloat64{1}(Float64(x))

@inline function MultiFloat64{2}(x::Int128)
    x0 = Float64(x)
    x1 = Float64(x - Int128(x0))
    MultiFloat64{2}((x0, x1))
end

@inline function MultiFloat64{2}(x::UInt128)
    x0 = Float64(x)
    x1 = Float64(reinterpret(Int128, x - UInt128(x0)))
    MultiFloat64{2}((x0, x1))
end

@inline function MultiFloat64{N}(x::Int64) where {N}
    x0 = Float64(x)
    x1 = Float64(x - Int64(x0))
    MultiFloat64{N}((x0, x1, ntuple(_ -> 0.0, N - 2)...))
end

@inline function MultiFloat64{N}(x::UInt64) where {N}
    x0 = Float64(x)
    x1 = Float64(reinterpret(Int64, x - UInt64(x0)))
    MultiFloat64{N}((x0, x1, ntuple(_ -> 0.0, N - 2)...))
end

@inline function MultiFloat64{N}(x::Int128) where {N}
    x0 = Float64(x)
    r1 = x - Int128(x0)
    x1 = Float64(r1)
    x2 = Float64(r1 - Int128(x1))
    MultiFloat64{N}((x0, x1, x2, ntuple(_ -> 0.0, N - 3)...))
end

@inline function MultiFloat64{N}(x::UInt128) where {N}
    x0 = Float64(x)
    r1 = reinterpret(Int128, x - UInt128(x0))
    x1 = Float64(r1)
    x2 = Float64(r1 - Int128(x1))
    MultiFloat64{N}((x0, x1, x2, ntuple(_ -> 0.0, N - 3)...))
end

################################################### CONVERSION FROM MULTIFLOAT64
############################################################# TO PRIMITIVE TYPES

import Base: Float16, Float32, Float64
@inline Float16(x::MultiFloat64{N}) where {N} = Float16(x.x[1])
@inline Float32(x::MultiFloat64{N}) where {N} = Float32(x.x[1])
@inline Float64(x::MultiFloat64{N}) where {N} = x.x[1]

##################################################### CONVERSION OF MULTIFLOAT64
########################################################## TO AND FROM BIG TYPES

import Base: BigFloat

BigFloat(x::MultiFloat64{N}) where {N} =
    +(ntuple(i -> BigFloat(x.x[N-i+1]), N)...)

function MultiFloat64{N}(x::BigFloat) where {N}
    setprecision(Int(precision(x))) do
        r = Vector{BigFloat}(undef, N)
        y = Vector{Float64}(undef, N)
        r[1] = x
        y[1] = Float64(r[1])
        for i = 2 : N
            r[i] = r[i-1] - y[i-1]
            y[i] = Float64(r[i])
        end
        MultiFloat64{N}((y...,))
    end
end

function MultiFloat64{N}(x::BigInt) where {N}
    y = Vector{Float64}(undef, N)
    for i = 1 : N
        y[i] = Float64(x)
        x -= BigInt(y[i])
    end
    MultiFloat64{N}((y...,))
end

function MultiFloat64{N}(x::Rational{T}) where {N, T}
    MultiFloat64{N}(numerator(x)) / MultiFloat64{N}(denominator(x))
end

################################################################### MULTIFLOAT64
################################################################ PROMOTION RULES

import Base: promote_rule

promote_rule(::Type{MultiFloat64{N}}, ::Type{Int8}) where {N} = MultiFloat64{N}
promote_rule(::Type{MultiFloat64{N}}, ::Type{Int16}) where {N} = MultiFloat64{N}
promote_rule(::Type{MultiFloat64{N}}, ::Type{Int32}) where {N} = MultiFloat64{N}
promote_rule(::Type{MultiFloat64{N}}, ::Type{Int64}) where {N} = MultiFloat64{N}
promote_rule(::Type{MultiFloat64{N}}, ::Type{Int128}) where {N} = MultiFloat64{N}
promote_rule(::Type{MultiFloat64{N}}, ::Type{Bool}) where {N} = MultiFloat64{N}
promote_rule(::Type{MultiFloat64{N}}, ::Type{UInt8}) where {N} = MultiFloat64{N}
promote_rule(::Type{MultiFloat64{N}}, ::Type{UInt16}) where {N} = MultiFloat64{N}
promote_rule(::Type{MultiFloat64{N}}, ::Type{UInt32}) where {N} = MultiFloat64{N}
promote_rule(::Type{MultiFloat64{N}}, ::Type{UInt64}) where {N} = MultiFloat64{N}
promote_rule(::Type{MultiFloat64{N}}, ::Type{UInt128}) where {N} = MultiFloat64{N}
promote_rule(::Type{MultiFloat64{N}}, ::Type{Float16}) where {N} = MultiFloat64{N}
promote_rule(::Type{MultiFloat64{N}}, ::Type{Float32}) where {N} = MultiFloat64{N}
promote_rule(::Type{MultiFloat64{N}}, ::Type{Float64}) where {N} = MultiFloat64{N}

################################################################################
################################################################################

import Base: zero, one, iszero, isone, signbit
@inline zero(::Type{MultiFloat64{N}}) where {N} = MultiFloat64{N}(0.0)
@inline one(::Type{MultiFloat64{N}}) where {N} = MultiFloat64{N}(1.0)
@inline iszero(x::MultiprecisionFloats.MultiFloat64{N}) where {N} =
    (&)(ntuple(i -> iszero(x.x[i]), N)...)
@inline isone(x::MultiprecisionFloats.MultiFloat64{N}) where {N} =
    isone(x.x[1]) & (&)(ntuple(i -> iszero(x.x[i + 1]), N - 1)...)
@inline signbit(x::MultiFloat64{N}) where {N} = signbit(x.x[1])

import Base: issubnormal, isfinite, isinf, isnan
@inline issubnormal(x::MultiprecisionFloats.MultiFloat64{N}) where {N} =
    issubnormal(x.x[1])
@inline isfinite(x::MultiprecisionFloats.MultiFloat64{N}) where {N} =
    isfinite(x.x[1])
@inline isinf(x::MultiprecisionFloats.MultiFloat64{N}) where {N} =
    isinf(x.x[1])
@inline isnan(x::MultiprecisionFloats.MultiFloat64{N}) where {N} =
    isnan(x.x[1])

################################################################################
################################################################################

import Base: +, -, *
@inline +(x::Float64, y::MultiFloat64{N}) where {N} = y + x
@inline -(x::MultiFloat64{N}) where {N} =
    MultiFloat64{N}(ntuple(i -> -x.x[i], N))
@inline -(x::MultiFloat64{N}, y::MultiFloat64{N}) where {N} = x + (-y)
@inline -(x::MultiFloat64{N}, y::Float64) where {N} = x + (-y)
@inline -(x::Float64, y::MultiFloat64{N}) where {N} = -(y + (-x))
@inline *(x::Float64, y::MultiFloat64{N}) where {N} = y * x

# TODO: Special-case these operations for a single operand.
import Base: inv, abs2
@inline inv(x::MultiFloat64{N}) where {N} = one(MultiFloat64{N}) / x
@inline abs2(x::MultiFloat64{N}) where {N} = x * x

# TODO: Add accurate comparison operators. Sloppy stop-gap operators for now.
import Base: ==, !=, <, >, <=, >=
@inline ==(x::MultiFloat64{N}, y::MultiFloat64{N}) where {N} =
    (x.x[1] == y.x[1]) & (x.x[2] == y.x[2])
@inline !=(x::MultiFloat64{N}, y::MultiFloat64{N}) where {N} =
    (x.x[1] != y.x[1]) | (x.x[2] != y.x[2])
@inline <(x::MultiFloat64{N}, y::MultiFloat64{N}) where {N} =
    (x.x[1] < y.x[1]) | ((x.x[1] == y.x[1]) & (x.x[2] < y.x[2]))
@inline >(x::MultiFloat64{N}, y::MultiFloat64{N}) where {N} =
    (x.x[1] > y.x[1]) | ((x.x[1] == y.x[1]) & (x.x[2] > y.x[2]))
@inline <=(x::MultiFloat64{N}, y::MultiFloat64{N}) where {N} =
    (x.x[1] < y.x[1]) | ((x.x[1] == y.x[1]) & (x.x[2] <= y.x[2]))
@inline >=(x::MultiFloat64{N}, y::MultiFloat64{N}) where {N} =
    (x.x[1] > y.x[1]) | ((x.x[1] == y.x[1]) & (x.x[2] >= y.x[2]))

import DZMisc: scale
@inline scale(a::Float64, x::MultiFloat64{N}) where {N} =
    MultiFloat64{N}(ntuple(i -> a * x.x[i], N))

import DZMisc: dbl
@inline dbl(x::MultiFloat64{N}) where {N} = scale(2.0, x)

################################################################################
################################################################################

import Base: +, *, /, sqrt

for i = 2 : 8
    eval(two_pass_renorm_func(i, sloppy=true))
    eval(MF64_alias(i))
    eval(MF64_add_func(i, sloppy=true))
    eval(MF64_F64_add_func(i, sloppy=true))
    eval(MF64_mul_func(i, sloppy=true))
    eval(MF64_F64_mul_func(i, sloppy=true))
    eval(MF64_div_func(i, sloppy=true))
    eval(MF64_sqrt_func(i, sloppy=true))
end

for (_, v) in MultiprecisionFloatCodeGenerator.MPADD_CACHE; eval(v); end

end # module MultiprecisionFloats
