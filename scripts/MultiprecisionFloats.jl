__precompile__(false)
module MultiprecisionFloats

using MultiprecisionFloatCodeGenerator

function two_sum(a::Float64, b::Float64)
    s = a + b
    v = s - a
    s, (a - (s - v)) + (b - v)
end

function quick_two_sum(a::Float64, b::Float64)
    s = a + b
    s, b - (s - a)
end

function two_prod(a::Float64, b::Float64)
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

MultiFloat64{N}(x::MultiFloat64{N}) where {N} = x

MultiFloat64{N}(x::Float64) where {N} =
    MultiFloat64{N}((x, ntuple(_ -> 0.0, N - 1)...))

# Values of the types Bool, Int8, UInt8, Int16, UInt16, Float16, Int32, UInt32,
# and Float32 can be converted losslessly to a single Float64, which has 53
# bits of integer precision.

MultiFloat64{N}(x::Bool   ) where {N} = MultiFloat64{N}(Float64(x))
MultiFloat64{N}(x::Int8   ) where {N} = MultiFloat64{N}(Float64(x))
MultiFloat64{N}(x::UInt8  ) where {N} = MultiFloat64{N}(Float64(x))
MultiFloat64{N}(x::Int16  ) where {N} = MultiFloat64{N}(Float64(x))
MultiFloat64{N}(x::UInt16 ) where {N} = MultiFloat64{N}(Float64(x))
MultiFloat64{N}(x::Float16) where {N} = MultiFloat64{N}(Float64(x))
MultiFloat64{N}(x::Int32  ) where {N} = MultiFloat64{N}(Float64(x))
MultiFloat64{N}(x::UInt32 ) where {N} = MultiFloat64{N}(Float64(x))
MultiFloat64{N}(x::Float32) where {N} = MultiFloat64{N}(Float64(x))

# Values of the types Int64, UInt64, Int128, and UInt128 cannot be converted
# losslessly to a single Float64 and must be split into multiple components.

MultiFloat64{1}(x::Int64  ) = MultiFloat64{1}(Float64(x))
MultiFloat64{1}(x::UInt64 ) = MultiFloat64{1}(Float64(x))
MultiFloat64{1}(x::Int128 ) = MultiFloat64{1}(Float64(x))
MultiFloat64{1}(x::UInt128) = MultiFloat64{1}(Float64(x))

function MultiFloat64{2}(x::Int128)
    x0 = Float64(x)
    x1 = Float64(x - Int128(x0))
    MultiFloat64{2}((x0, x1))
end

function MultiFloat64{2}(x::UInt128)
    x0 = Float64(x)
    x1 = Float64(reinterpret(Int128, x - UInt128(x0)))
    MultiFloat64{2}((x0, x1))
end

function MultiFloat64{N}(x::Int64) where {N}
    x0 = Float64(x)
    x1 = Float64(x - Int64(x0))
    MultiFloat64{N}((x0, x1, ntuple(_ -> 0.0, N - 2)...))
end

function MultiFloat64{N}(x::UInt64) where {N}
    x0 = Float64(x)
    x1 = Float64(reinterpret(Int64, x - UInt64(x0)))
    MultiFloat64{N}((x0, x1, ntuple(_ -> 0.0, N - 2)...))
end

function MultiFloat64{N}(x::Int128) where {N}
    x0 = Float64(x)
    r1 = x - Int128(x0)
    x1 = Float64(r1)
    x2 = Float64(r1 - Int128(x1))
    MultiFloat64{N}((x0, x1, x2, ntuple(_ -> 0.0, N - 3)...))
end

function MultiFloat64{N}(x::UInt128) where {N}
    x0 = Float64(x)
    r1 = reinterpret(Int128, x - UInt128(x0))
    x1 = Float64(r1)
    x2 = Float64(r1 - Int128(x1))
    MultiFloat64{N}((x0, x1, x2, ntuple(_ -> 0.0, N - 3)...))
end

##################################################### CONVERSION OF MULTIFLOAT64
########################################################### TO AND FROM BIGFLOAT

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

################################################### CONVERSION FROM MULTIFLOAT64
############################################################# TO PRIMITIVE TYPES

import Base: Float16, Float32, Float64

Float16(x::MultiFloat64{N}) where {N} = Float16(x.x[1])
Float32(x::MultiFloat64{N}) where {N} = Float32(x.x[1])
Float64(x::MultiFloat64{N}) where {N} = x.x[1]

################################################################################
################################################################################

import Base: zero, one, signbit

zero(::Type{MultiFloat64{N}}) where {N} = MultiFloat64{N}(0.0)
one(::Type{MultiFloat64{N}}) where {N} = MultiFloat64{N}(1.0)
signbit(x::MultiFloat64{N}) where {N} = signbit(x.x[1])

################################################################################
################################################################################

import Base: +, -, *
+(x::Float64, y::MultiFloat64{N}) where {N} = y + x
-(x::MultiFloat64{N}) where {N} = MultiFloat64{N}(ntuple(i -> -x.x[i], N))
-(x::MultiFloat64{N}, y::MultiFloat64{N}) where {N} = x + (-y)
-(x::MultiFloat64{N}, y::Float64) where {N} = x + (-y)
-(x::Float64, y::MultiFloat64{N}) where {N} = -(y + (-x))
*(x::Float64, y::MultiFloat64{N}) where {N} = y * x

scale(a::Float64, x::MultiFloat64{N}) where {N} =
    MultiFloat64{N}(ntuple(i -> a * x.x[i], N))

import DZMisc: dbl
dbl(x::MultiFloat64{N}) where {N} = scale(2.0, x)

################################################################################
################################################################################

import Base: +, *, /, sqrt

for i = 2 : 4
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

export Float64x2, Float64x3, Float64x4

end # module MultiprecisionFloats
