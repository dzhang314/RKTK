__precompile__(false)
module RKTK

export RKOCEvaluator, evaluate_residual!, evaluate_jacobian!

using DZMisc: dbl, rooted_tree_count
using RKTKData: GAMMA, SIZE_DEFICIT, TOTAL_SIZE_DEFICIT, OPCODES

struct RKOCEvaluator{T <: Real}
    order::Int
    num_stages::Int
    num_vars::Int
    num_constrs::Int
    inv_gamma::Vector{T}
    u::Vector{T}
    v::Vector{T}
end

function RKOCEvaluator{T}(order::Int, num_stages::Int) where {T <: Real}
    num_vars = div(num_stages * (num_stages + 1), 2)
    num_constrs = rooted_tree_count(order)
    inv_gamma = inv.(T.(GAMMA))
    table_deficit = TOTAL_SIZE_DEFICIT[num_constrs]
    table_size = (num_constrs - 1) * num_stages - table_deficit
    u = Vector{T}(undef, table_size)
    v = Vector{T}(undef, table_size)
    RKOCEvaluator{T}(order, num_stages, num_vars, num_constrs, inv_gamma, u, v)
end

function populate_u_init!(evaluator::RKOCEvaluator{T}, x::Vector{T}) where {T <: Real}
    k = 1
    for i = 1 : evaluator.num_stages - 1
        @inbounds evaluator.u[i] = x[k]
        for j = 1 : i - 1
            @inbounds evaluator.u[i] += x[k + j]
        end
        k += i
    end
end

function populate_v_init!(evaluator::RKOCEvaluator{T}, x::Vector{T}, var_idx::Int) where {T <: Real}
    j = 0
    for i = 1 : evaluator.num_stages - 1
        @inbounds evaluator.v[i] = (j == var_idx)
        j += 1
        for _ = 1 : i - 1
            @inbounds evaluator.v[i] += (j == var_idx)
            j += 1
        end
    end
end

@inline function dot_inplace(n::Int, v::Vector{T}, v_offset::Int, w::Vector{T}, w_offset::Int) where {T <: Real}
    @inbounds result = v[v_offset + 1] * w[w_offset + 1]
    @simd for i = 2 : n
        @inbounds result += v[v_offset + i] * w[w_offset + i]
    end
    result
end

@inline subarray_offset(evaluator::RKOCEvaluator{T}, i::Int) where {T <: Real} =
    @inbounds evaluator.num_stages * i - TOTAL_SIZE_DEFICIT[i + 1]

function populate_u!(evaluator::RKOCEvaluator{T}, x::Vector{T}) where {T <: Real}
    populate_u_init!(evaluator, x)
    pos = evaluator.num_stages - 1
    u = evaluator.u # local alias for convenience
    for constr_idx = 3 : evaluator.num_constrs
        @inbounds n = evaluator.num_stages - SIZE_DEFICIT[constr_idx - 1]
        @inbounds op, a, b = OPCODES[constr_idx - 2]
        if op
            ao = subarray_offset(evaluator, a)
            skip_idx = evaluator.num_stages - n
            j = div(skip_idx * (skip_idx + 1), 2) - 1
            for i = 1 : n
                @inbounds u[pos+i] = dot_inplace(i, x, j, u, ao)
                j += skip_idx
                skip_idx += 1
            end
        elseif a == b
            ao = subarray_offset(evaluator, a)
            @simd ivdep for i = 1 : n
                @inbounds u[pos+i] = abs2(u[ao+i])
            end
        else
            @inbounds ad = SIZE_DEFICIT[a + 1]
            @inbounds bd = SIZE_DEFICIT[b + 1]
            md = max(ad, bd)
            oa = subarray_offset(evaluator, a) + md - ad
            ob = subarray_offset(evaluator, b) + md - bd
            @simd ivdep for i = 1 : n
                @inbounds u[pos+i] = u[oa+i] * u[ob+i]
            end
        end
        pos += n
    end
end

function populate_v!(evaluator::RKOCEvaluator{T}, x::Vector{T}, var_idx::Int) where {T <: Real}
    populate_v_init!(evaluator, x, var_idx)
    pos = evaluator.num_stages - 1
    u = evaluator.u
    v = evaluator.v
    for constr_idx = 3 : evaluator.num_constrs
        @inbounds n = evaluator.num_stages - SIZE_DEFICIT[constr_idx - 1]
        @inbounds op, a, b = OPCODES[constr_idx - 2]
        if op
            ao = subarray_offset(evaluator, a)
            skip_idx = evaluator.num_stages - n
            j = div(skip_idx * (skip_idx + 1), 2) - 1
            for i = 1 : n
                @inbounds v[pos+i] = dot_inplace(i, x, j, v, ao)
                if j <= var_idx < j + i
                    @inbounds v[pos+i] += u[ao + var_idx - j + 1]
                end
                j += skip_idx
                skip_idx += 1
            end
        elseif a == b
            ao = subarray_offset(evaluator, a)
            @simd ivdep for i = 1 : n
                @inbounds v[pos+i] = dbl(v[ao+i] * u[ao+i])
            end
        else
            @inbounds ad = SIZE_DEFICIT[a + 1]
            @inbounds bd = SIZE_DEFICIT[b + 1]
            md = max(ad, bd)
            ao = subarray_offset(evaluator, a) + md - ad
            bo = subarray_offset(evaluator, b) + md - bd
            @simd ivdep for i = 1 : n
                @inbounds v[pos+i] = v[ao+i] * u[bo+i] + u[ao+i] * v[bo+i]
            end
        end
        pos += n
    end
end

function evaluate_residual!(res::Vector{T}, x::Vector{T}, evaluator::RKOCEvaluator{T}) where {T <: Real}
    @inbounds res[1] = -one(T)
    b_idx = evaluator.num_vars - evaluator.num_stages + 1
    @simd for i = b_idx : evaluator.num_vars
        @inbounds res[1] += x[i]
    end
    populate_u!(evaluator, x)
    pos = 0
    for constr_idx = 2 : evaluator.num_constrs
        @inbounds n = evaluator.num_stages - SIZE_DEFICIT[constr_idx - 1]
        @inbounds res[constr_idx] = dot_inplace(n, evaluator.u, pos, x, evaluator.num_vars - n)
        pos += n
    end
    @simd ivdep for constr_idx = 2 : evaluator.num_constrs
        @inbounds res[constr_idx] -= evaluator.inv_gamma[constr_idx - 1]
    end
end

function evaluate_jacobian!(jac::Matrix{T}, x::Vector{T}, evaluator::RKOCEvaluator{T}) where {T <: Real}
    populate_u!(evaluator, x)
    for var_idx = 1 : evaluator.num_vars
        populate_v!(evaluator, x, var_idx - 1)
        @inbounds jac[1, var_idx] = (var_idx + evaluator.num_stages > evaluator.num_vars)
        pos = 0
        for constr_idx = 2 : evaluator.num_constrs
            @inbounds n = evaluator.num_stages - SIZE_DEFICIT[constr_idx - 1]
            m = evaluator.num_vars - n
            @inbounds jac[constr_idx, var_idx] = dot_inplace(n, evaluator.v, pos, x, m)
            if var_idx + n > evaluator.num_vars
                @inbounds jac[constr_idx, var_idx] += evaluator.u[pos + var_idx - m]
            end
            pos += n
        end
    end
end

end # module RKTK
