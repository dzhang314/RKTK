__precompile__(false)
module RKTK

export RKOCEvaluator, evaluate_residuals!, evaluate_jacobian!

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

function populate_u_init!(evaluator::RKOCEvaluator{T},
        x::Vector{T}) where {T <: Real}
    k = 1
    for i = 1 : evaluator.num_stages - 1
        @inbounds evaluator.u[i] = x[k]
        for j = 1 : i - 1
            @inbounds evaluator.u[i] += x[k + j]
        end
        k += i
    end
end

function populate_v_init!(evaluator::RKOCEvaluator{T},
        x::Vector{T}, var_index::Int) where {T <: Real}
    j = 0
    for i = 1 : evaluator.num_stages - 1
        @inbounds evaluator.v[i] = (j == var_index)
        j += 1
        for _ = 1 : i - 1
            @inbounds evaluator.v[i] += (j == var_index)
            j += 1
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

@inline function subarray_offset(
        evaluator::RKOCEvaluator{T}, i::Int) where {T <: Real}
    @inbounds evaluator.num_stages * i - TOTAL_SIZE_DEFICIT[i + 1]
end

function populate_u!(evaluator::RKOCEvaluator{T},
        x::Vector{T}) where {T <: Real}
    populate_u_init!(evaluator, x)
    pos = evaluator.num_stages - 1
    u = evaluator.u # local alias for convenience
    for op_index = 2 : evaluator.num_constrs - 1
        @inbounds n = evaluator.num_stages - SIZE_DEFICIT[op_index]
        @inbounds op, a, b = OPCODES[op_index - 1]
        if op
            ao = subarray_offset(evaluator, a)
            skp = evaluator.num_stages - n
            idex = div(skp * (skp + 1), 2) - 1
            for i = 1 : n
                @inbounds u[pos+i] = dot_inplace(i, x, idex, u, ao)
                idex += skp
                skp += 1
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

function populate_v!(evaluator::RKOCEvaluator{T},
        x::Vector{T}, var_index::Int) where {T <: Real}
    populate_v_init!(evaluator, x, var_index)
    pos = evaluator.num_stages - 1
    u = evaluator.u
    v = evaluator.v
    for j = 2 : evaluator.num_constrs - 1
        @inbounds n = evaluator.num_stages - SIZE_DEFICIT[j]
        @inbounds op, a, b = OPCODES[j - 1]
        if op
            ao = subarray_offset(evaluator, a)
            skp = evaluator.num_stages - n
            idex = div(skp * (skp + 1), 2) - 1
            for i = 1 : n
                @inbounds v[pos+i] = dot_inplace(i, x, idex, v, ao)
                if idex <= var_index < idex + i
                    @inbounds v[pos+i] += u[ao + var_index - idex + 1]
                end
                idex += skp
                skp += 1
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

function evaluate_residuals!(res::Vector{T},
        x::Vector{T}, evaluator::RKOCEvaluator{T}) where {T <: Real}
    res[1] = -one(T)
    b_index = evaluator.num_vars - evaluator.num_stages + 1
    @simd for i = b_index : evaluator.num_vars
        @inbounds res[1] += x[i]
    end
    populate_u!(evaluator, x)
    j = 2
    pos = 0
    for i = 1 : evaluator.num_constrs - 1
        n = evaluator.num_stages - SIZE_DEFICIT[i]
        res[j] = dot_inplace(n, evaluator.u, pos, x, evaluator.num_vars - n)
        res[j] -= evaluator.inv_gamma[i]
        j += 1
        pos += n
    end
end

function evaluate_jacobian!(jac::Matrix{T},
        x::Vector{T}, evaluator::RKOCEvaluator{T}) where {T <: Real}
    k = 1
    populate_u!(evaluator, x)
    for var_index = 1 : evaluator.num_vars
        populate_v!(evaluator, x, var_index - 1)
        jac[k] = (var_index + evaluator.num_stages > evaluator.num_vars)
        k += 1
        pos = 0
        for j = 1 : evaluator.num_constrs - 1
            n = evaluator.num_stages - SIZE_DEFICIT[j]
            m = evaluator.num_vars - n
            jac[k] = dot_inplace(n, evaluator.v, pos, x, m)
            if var_index + n > evaluator.num_vars
                jac[k] += evaluator.u[pos + var_index - m]
            end
            k += 1
            pos += n
        end
    end
end

end # module RKTK
