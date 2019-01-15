__precompile__(false)
module RKTKParallel

include("rounds/RK-ROUNDS-5-6.jl")

export ORDER, NUM_STAGES, NUM_VARS, NUM_CONSTRS, RKOCEvaluatorParallel,
    evaluate_residual_parallel!, evaluate_jacobian_parallel!

using Base.Threads: @threads, nthreads, threadid

using DZMisc: dbl
using RKTKData: GAMMA

struct RKOCEvaluatorParallel{T <: Real}
    inv_gamma::Vector{T}
    u::Vector{T}
    vs::Vector{Vector{T}}
end

function RKOCEvaluatorParallel{T}() where {T <: Real}
    inv_gamma = inv.(T.(GAMMA))
    u = Vector{T}(undef, TABLE_SIZE)
    vs = [Vector{T}(undef, TABLE_SIZE) for _ = 1 : nthreads()]
    RKOCEvaluatorParallel{T}(inv_gamma, u, vs)
end

function populate_u_init!(u::Vector{T}, x::Vector{T}) where {T <: Real}
    k = 1
    for i = 1 : NUM_STAGES - 1
        @inbounds result = x[k]
        @simd for j = 1 : i - 1
            @inbounds result += x[k + j]
        end
        @inbounds u[i] = result
        k += i
    end
end

function populate_v_init!(v::Vector{T}, var_index::Int) where {T <: Real}
    j = 0
    for i = 1 : NUM_STAGES - 1
        result = (j == var_index)
        j += 1
        for _ = 1 : i - 1
            result |= (j == var_index)
            j += 1
        end
        @inbounds v[i] = T(result)
    end
end

function lvm_u!(m::Vector{T}, dst_begin::Int, dst_end::Int,
                x::Vector{T}, src_begin::Int) where {T <: Real}
    dst_size = dst_end - dst_begin + 1
    skip = NUM_STAGES - dst_size
    index = div(skip * (skip + 1), 2)
    @inbounds m[dst_begin] = x[index] * m[src_begin]
    for i = 1 : dst_size - 1
        index += skip
        skip += 1
        @inbounds result = x[index] * m[src_begin]
        @simd for j = 1 : i
            @inbounds result += x[index + j] * m[src_begin + j]
        end
        @inbounds m[dst_begin + i] = result
    end
end

function lvm_v!(v::Vector{T}, var_index::Int,
                u::Vector{T}, dst_begin::Int, dst_end::Int,
                x::Vector{T}, src_begin::Int) where {T <: Real}
    dst_size = dst_end - dst_begin + 1
    skip = NUM_STAGES - dst_size
    index = div(skip * (skip + 1), 2)
    if index == var_index + 1
        @inbounds v[dst_begin] = (x[index - 1 + 1] * v[src_begin] +
            u[src_begin + var_index - index + 1])
    else
        @inbounds v[dst_begin] = (x[index - 1 + 1] * v[src_begin])
    end
    for i = 1 : dst_size - 1
        index += skip
        skip += 1
        @inbounds result = x[index] * v[src_begin]
        @simd for j = 1 : i
            @inbounds result += x[index + j] * v[src_begin + j]
        end
        if (index <= var_index + 1) && (var_index + 1 <= index + i)
            @inbounds result += u[src_begin + var_index - index + 1]
        end
        @inbounds v[dst_begin + i] = result
    end
end

function populate_u_parallel!(u::Vector{T}, x::Vector{T}) where {T <: Real}
    populate_u_init!(u, x)
    for round in ROUNDS
        @threads for (_, dst_begin, dst_end, src1, src2) in round
            if src2 == -1
                lvm_u!(u, dst_begin, dst_end, x, src1)
            elseif src1 == src2
                @simd ivdep for i = 0 : dst_end - dst_begin
                    @inbounds u[dst_begin + i] = u[src1 + i]^2
                end
            else
                @simd ivdep for i = 0 : dst_end - dst_begin
                    @inbounds u[dst_begin + i] = u[src1 + i] * u[src2 + i]
                end
            end
        end
    end
end

function populate_v_parallel!(v::Vector{T}, u::Vector{T},
                              x::Vector{T}, var_index::Int) where {T <: Real}
    populate_v_init!(v, var_index)
    for round in ROUNDS
        for (_, dst_begin, dst_end, src1, src2) in round
            if src2 == -1
                lvm_v!(v, var_index, u, dst_begin, dst_end, x, src1)
            elseif src1 == src2
                @simd ivdep for i = 0 : dst_end - dst_begin
                    @inbounds v[dst_begin + i] = dbl(u[src1 + i] * v[src1 + i])
                end
            else
                @simd ivdep for i = 0 : dst_end - dst_begin
                    @inbounds v[dst_begin + i] = (u[src1 + i] * v[src2 + i] +
                                                    u[src2 + i] * v[src1 + i])
                end
            end
        end
    end
end

@inline function dot_inplace(n::Int,
        v::Vector{T}, v_offset::Int,
        w::Vector{T}, w_offset::Int) where {T <: Real}
    @inbounds result = v[v_offset + 1] * w[w_offset + 1]
    @simd for i = 2 : n
        @inbounds result += v[v_offset + i] * w[w_offset + i]
    end
    result
end

function evaluate_residual_parallel!(res::Vector{T}, x::Vector{T},
        evaluator::RKOCEvaluatorParallel{T}) where {T <: Real}
    u = evaluator.u
    inv_gamma = evaluator.inv_gamma
    populate_u_parallel!(u, x)
    let
        first = -one(T)
        b_idx = NUM_VARS - NUM_STAGES + 1
        @simd for i = b_idx : NUM_VARS
            @inbounds first += x[i]
        end
        @inbounds res[1] = first
    end
    @inbounds res[2] = dot_inplace(NUM_STAGES - 1, u, 0,
                                   x, NUM_VARS - NUM_STAGES + 1) - T(0.5)
    for round in ROUNDS
        @threads for (res_index, dst_begin, dst_end, _, _) in round
            j = dst_begin - 1
            n = dst_end - j
            @inbounds res[res_index] = (dot_inplace(n, u, j, x, NUM_VARS - n) -
                                        inv_gamma[res_index - 1])
        end
    end
end

function evaluate_jacobian_parallel!(jac::Matrix{T}, x::Vector{T},
        evaluator::RKOCEvaluatorParallel{T}) where {T <: Real}
    u = evaluator.u
    populate_u_parallel!(u, x)
    @threads for var_idx = 1 : NUM_VARS
        @inbounds v = evaluator.vs[threadid()]
        populate_v_parallel!(v, u, x, var_idx - 1)
        @inbounds jac[1, var_idx] = T(var_idx + NUM_STAGES > NUM_VARS)
        let
            n = NUM_STAGES - 1
            m = NUM_VARS - n
            result = dot_inplace(n, v, 0, x, m)
            if var_idx + n > NUM_VARS
                @inbounds result += u[var_idx - m]
            end
            @inbounds jac[2, var_idx] = result
        end
        for round in ROUNDS
            for (res_index, dst_begin, dst_end, _, _) in round
                n = dst_end - dst_begin + 1
                m = NUM_VARS - n
                result = dot_inplace(n, v, dst_begin - 1, x, m)
                if var_idx + n > NUM_VARS
                    @inbounds result += u[dst_begin - 1 + var_idx - m]
                end
                @inbounds jac[res_index, var_idx] = result
            end
        end
    end
end

end # module RKTKParallel
