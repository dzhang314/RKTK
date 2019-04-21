module RKTK2

export RKOCEvaluator, evaluate_residual!, evaluate_jacobian!,
    evaluate_error_coefficients!, evaluate_error_jacobian!,
    constrain!, compute_order!, compute_stages,
    rk4_table, extrapolated_euler_table, rkck5_table, dopri5_table, rkf8_table,
    RKSolver, runge_kutta_step!,
    RKOCBackpropEvaluator, evaluate_residual2, evaluate_gradient!,
    populate_explicit!#, RKOCBackpropFSGDOptimizer, step!

using Base.Threads: @threads, nthreads, threadid
using LinearAlgebra: mul!, ldiv!, qrfactUnblocked!

using DZMisc: rmk, say, dbl, norm2, normalize!,
    RootedTree, rooted_tree_count, rooted_trees,
    butcher_density, butcher_symmetry

################################################################################

function Base.show(io::IO, tree::RootedTree)::Nothing
    print(io, '[')
    for (subtree, multiplicity) in tree.children
        print(io, subtree)
        if multiplicity != 1
            print(io, '^', multiplicity)
        end
    end
    print(io, ']')
end

function tree_index_table(
        trees_of_order::Vector{Vector{RootedTree}})::Dict{String,Int}
    order = length(trees_of_order)
    tree_index = Dict{String,Int}()
    i = 0
    for p = 1 : order - 1, tree in trees_of_order[p]
        tree_index[string(tree)] = (i += 1)
    end
    tree_index
end

function dependency_table(
        trees_of_order::Vector{Vector{RootedTree}})::Vector{Vector{Int}}
    order = length(trees_of_order)
    tree_index = tree_index_table(trees_of_order)
    dependencies = Vector{Int}[]
    for p = 1 : order, tree in trees_of_order[p]
        num_children = length(tree.children)
        if num_children == 0
            push!(dependencies, Int[])
        elseif num_children == 1
            child, m = tree.children[1]
            if m == 1
                a = tree_index[string(child)]
                push!(dependencies, [a])
            elseif m % 2 == 0
                half_child = RootedTree(0, [child => div(m, 2)])
                a = tree_index[string(half_child)]
                push!(dependencies, [a, a])
            else
                half_child = RootedTree(0, [child => div(m, 2)])
                half_plus_one_child = RootedTree(0, [child => div(m, 2) + 1])
                a = tree_index[string(half_child)]
                b = tree_index[string(half_plus_one_child)]
                push!(dependencies, [a, b])
            end
        else
            head_factor = RootedTree(0, tree.children[1 : 1])
            tail_factor = RootedTree(0, tree.children[2 : end])
            a = tree_index[string(head_factor)]
            b = tree_index[string(tail_factor)]
            push!(dependencies, [a, b])
        end
    end
    dependencies
end

function generation_table(dependencies::Vector{Vector{Int}})::Vector{Int}
    generation = Vector{Int}(undef, length(dependencies))
    for (i, deps) in enumerate(dependencies)
        if length(deps) == 0
            generation[i] = 0
        else
            generation[i] = maximum(generation[j] for j in deps) + 1
        end
    end
    generation
end

function deficit_table(dependencies::Vector{Vector{Int}})::Vector{Int}
    deficit = Vector{Int}(undef, length(dependencies))
    for (i, deps) in enumerate(dependencies)
        if length(deps) == 0
            deficit[i] = 0
        elseif length(deps) == 1
            deficit[i] = deficit[deps[1]] + 1
        else
            deficit[i] = maximum(deficit[j] for j in deps)
        end
    end
    deficit
end

function range_table(dependencies::Vector{Vector{Int}},
        num_stages::Int)::Vector{Tuple{Int,Int}}
    deficit = deficit_table(dependencies)
    lengths = num_stages .- deficit
    clamp!(lengths, 0, typemax(Int))
    lengths[1] = 0
    cumsum!(lengths, lengths)
    table_size = lengths[end]
    ranges = vcat([(1, 0)], collect(
        zip(lengths[1 : end - 1] .+ 1, lengths[2 : end])))
end

function instruction_table(dependencies::Vector{Vector{Int}},
        ranges::Vector{Tuple{Int,Int}})::Vector{NTuple{5,Int}}
    instructions = Vector{NTuple{5,Int}}(undef, length(dependencies))
    for (i, deps) in enumerate(dependencies)
        num_deps = length(deps)
        a, b = ranges[i]
        if num_deps == 0
            instructions[i] = (i, a, b, -1, -1)
        elseif num_deps == 1
            c, d = ranges[deps[1]]
            instructions[i] = (i, a, b, c, -1)
        else
            c, d = ranges[deps[1]]
            e, f = ranges[deps[2]]
            instructions[i] = (i, a, b, d - (b - a), f - (b - a))
        end
    end
    instructions
end

################################################################################

function require!(required::Vector{Bool},
        dependencies::Vector{Vector{Int}}, index::Int)::Nothing
    if !required[index]
        required[index] = true
        for dep in dependencies[index]
            require!(required, dependencies, dep)
        end
    end
end

function assert_requirements_satisfied(required::Vector{Bool},
        dependencies::Vector{Vector{Int}}, indices::Vector{Int})::Nothing
    @assert(all(required[indices]))
    for index = 1 : length(dependencies)
        if required[index]
            @assert(all(required[dep] for dep in dependencies[index]))
        end
    end
end

function included_indices(mask::Vector{Bool})::Vector{Int}
    result = Vector{Int}(undef, sum(mask))
    count = 0
    for index = 1 : length(mask)
        if mask[index]
            result[count += 1] = index
        end
    end
    result
end

function restricted_indices(mask::Vector{Bool})::Vector{Int}
    result = Vector{Int}(undef, length(mask))
    count = 0
    for index = 1 : length(mask)
        if mask[index]
            result[index] = (count += 1)
        else
            result[index] = 0
        end
    end
    result
end

function inverse_index_table(indices::Vector{Int}, n::Int)
    result = [0 for _ = 1 : n]
    for (i, j) in enumerate(indices)
        result[j] = i
    end
    result
end

function restricted_trees_dependencies(indices::Vector{Int})
    max_index = maximum(indices)
    order = 1
    while max_index > rooted_tree_count(order)
        order += 1
    end
    trees_of_order = rooted_trees(order)
    full_dependencies = dependency_table(trees_of_order)
    required = [false for _ = 1 : length(full_dependencies)]
    for index in indices
        require!(required, full_dependencies, index)
    end
    assert_requirements_satisfied(required, full_dependencies, indices)
    included = included_indices(required)
    restrict = restricted_indices(required)
    restricted_trees = vcat(trees_of_order...)[included]
    restricted_dependencies = Vector{Int}[]
    for index in included
        push!(restricted_dependencies, restrict[full_dependencies[index]])
    end
    (restricted_trees, restricted_dependencies,
        inverse_index_table(restrict[indices], length(included)))
end

################################################################################

struct RKOCEvaluator{T <: Real}
    order::Int
    num_stages::Int
    num_vars::Int
    num_constrs::Int
    output_indices::Vector{Int}
    rounds::Vector{Vector{NTuple{5,Int}}}
    inv_density::Vector{T}
    inv_symmetry::Vector{T}
    u::Vector{T}
    vs::Vector{Vector{T}}
end

function RKOCEvaluator{T}(order::Int, num_stages::Int) where {T <: Real}
    # TODO: This assertion should not be necessary.
    @assert(order >= 2)
    num_vars = div(num_stages * (num_stages + 1), 2)
    trees_of_order = rooted_trees(order)
    num_constrs = sum(length.(trees_of_order))
    dependencies = dependency_table(trees_of_order)
    generation = generation_table(dependencies)
    ranges = range_table(dependencies, num_stages)
    table_size = ranges[end][2]
    instructions = instruction_table(dependencies, ranges)
    rounds = [NTuple{5,Int}[] for _ = 1 : maximum(generation) - 1]
    for (i, g) in enumerate(generation)
        if g > 1; push!(rounds[g - 1], instructions[i]); end
    end
    RKOCEvaluator{T}(order, num_stages, num_vars, num_constrs, Int[], rounds,
        inv.(T.(butcher_density.(vcat(trees_of_order...)))),
        inv.(T.(butcher_symmetry.(vcat(trees_of_order...)))),
        Vector{T}(undef, table_size),
        [Vector{T}(undef, table_size) for _ = 1 : nthreads()])
end

function RKOCEvaluator{T}(indices::Vector{Int},
        num_stages::Int) where {T <: Real}
    num_vars = div(num_stages * (num_stages + 1), 2)
    trees, dependencies, output_indices =
        restricted_trees_dependencies(indices)
    num_constrs = length(indices)
    generation = generation_table(dependencies)
    ranges = range_table(dependencies, num_stages)
    table_size = ranges[end][2]
    instructions = instruction_table(dependencies, ranges)
    rounds = [NTuple{5,Int}[] for _ = 1 : maximum(generation) - 1]
    for (i, g) in enumerate(generation)
        if g > 1; push!(rounds[g - 1], instructions[i]); end
    end
    RKOCEvaluator{T}(0, num_stages, num_vars, num_constrs,
        output_indices, rounds,
        inv.(T.(butcher_density.(trees))),
        inv.(T.(butcher_symmetry.(trees))),
        Vector{T}(undef, table_size),
        [Vector{T}(undef, table_size) for _ = 1 : nthreads()])
end

################################################################################

function populate_u_init!(evaluator::RKOCEvaluator{T},
        x::Vector{T})::Nothing where {T <: Real}
    u = evaluator.u
    k = 1
    for i = 1 : evaluator.num_stages - 1
        @inbounds result = x[k]
        @simd for j = 1 : i - 1
            @inbounds result += x[k + j]
        end
        @inbounds u[i] = result
        k += i
    end
end

function lvm_u!(evaluator::RKOCEvaluator{T}, dst_begin::Int, dst_end::Int,
                x::Vector{T}, src_begin::Int)::Nothing where {T <: Real}
    if dst_begin > dst_end; return; end
    u = evaluator.u
    dst_size = dst_end - dst_begin + 1
    skip = evaluator.num_stages - dst_size
    index = div(skip * (skip + 1), 2)
    @inbounds u[dst_begin] = x[index] * u[src_begin]
    for i = 1 : dst_size - 1
        index += skip
        skip += 1
        @inbounds result = x[index] * u[src_begin]
        @simd for j = 1 : i
            @inbounds result += x[index + j] * u[src_begin + j]
        end
        @inbounds u[dst_begin + i] = result
    end
end

function populate_u!(evaluator::RKOCEvaluator{T},
        x::Vector{T})::Nothing where {T <: Real}
    u = evaluator.u
    populate_u_init!(evaluator, x)
    for round in evaluator.rounds
        @threads for (_, dst_begin, dst_end, src1, src2) in round
            if src2 == -1
                lvm_u!(evaluator, dst_begin, dst_end, x, src1)
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

function populate_v_init!(evaluator::RKOCEvaluator{T},
        v::Vector{T}, var_index::Int)::Nothing where {T <: Real}
    j = 0
    for i = 1 : evaluator.num_stages - 1
        result = (j == var_index)
        j += 1
        for _ = 1 : i - 1
            result |= (j == var_index)
            j += 1
        end
        @inbounds v[i] = T(result)
    end
end

function lvm_v!(evaluator::RKOCEvaluator{T}, v::Vector{T}, var_index::Int,
                dst_begin::Int, dst_end::Int,
                x::Vector{T}, src_begin::Int)::Nothing where {T <: Real}
    if dst_begin > dst_end; return; end
    u = evaluator.u
    dst_size = dst_end - dst_begin + 1
    skip = evaluator.num_stages - dst_size
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

function populate_v!(evaluator::RKOCEvaluator{T}, v::Vector{T},
                     x::Vector{T}, var_index::Int)::Nothing where {T <: Real}
    u = evaluator.u
    populate_v_init!(evaluator, v, var_index)
    for round in evaluator.rounds
        for (_, dst_begin, dst_end, src1, src2) in round
            if src2 == -1
                lvm_v!(evaluator, v, var_index, dst_begin, dst_end, x, src1)
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
        w::Vector{T}, w_offset::Int)::T where {T <: Real}
    @inbounds result = zero(T)
    @simd for i = 1 : n
        @inbounds result += v[v_offset + i] * w[w_offset + i]
    end
    result
end

################################################################################

function evaluate_residual!(res::Vector{T}, x::Vector{T},
        evaluator::RKOCEvaluator{T})::Nothing where {T <: Real}
    @assert(length(x) == evaluator.num_vars)
    u = evaluator.u
    num_stages = evaluator.num_stages
    num_vars = evaluator.num_vars
    output_indices = evaluator.output_indices
    inv_density = evaluator.inv_density
    populate_u!(evaluator, x)
    if length(output_indices) == 0
        let
            first = -one(T)
            b_idx = num_vars - num_stages + 1
            @simd for i = b_idx : num_vars
                @inbounds first += x[i]
            end
            @inbounds res[1] = first
        end
        @inbounds res[2] = dot_inplace(num_stages - 1, u, 0,
                                       x, num_vars - num_stages + 1) - T(0.5)
        for round in evaluator.rounds
            @threads for (res_index, dst_begin, dst_end, _, _) in round
                j = dst_begin - 1
                n = dst_end - j
                @inbounds res[res_index] =
                    dot_inplace(n, u, j, x, num_vars - n) -
                    inv_density[res_index]
            end
        end
    else
        let
            output_index = output_indices[1]
            if output_index > 0
                first = -one(T)
                b_idx = num_vars - num_stages + 1
                @simd for i = b_idx : num_vars
                    @inbounds first += x[i]
                end
                @inbounds res[output_index] = first
            end
        end
        let
            output_index = output_indices[2]
            if output_index > 0
                b_idx = num_vars - num_stages + 1
                @inbounds res[output_index] =
                    dot_inplace(num_stages - 1, u, 0, x, b_idx) - T(0.5)
            end
        end
        for round in evaluator.rounds
            @threads for (res_index, dst_begin, dst_end, _, _) in round
                output_index = output_indices[res_index]
                if output_index > 0
                    j = dst_begin - 1
                    n = dst_end - j
                    @inbounds res[output_index] =
                        dot_inplace(n, u, j, x, num_vars - n) -
                        inv_density[res_index]
                end
            end
        end
    end
end

function evaluate_error_coefficients!(res::Vector{T}, x::Vector{T},
        evaluator::RKOCEvaluator{T})::Nothing where {T <: Real}
    @assert(length(x) == evaluator.num_vars)
    u = evaluator.u
    num_stages = evaluator.num_stages
    num_vars = evaluator.num_vars
    output_indices = evaluator.output_indices
    inv_density = evaluator.inv_density
    inv_symmetry = evaluator.inv_symmetry
    populate_u!(evaluator, x)
    if length(output_indices) == 0
        let
            first = -one(T)
            b_idx = num_vars - num_stages + 1
            @simd for i = b_idx : num_vars
                @inbounds first += x[i]
            end
            @inbounds res[1] = first
        end
        @inbounds res[2] = dot_inplace(num_stages - 1, u, 0,
                                       x, num_vars - num_stages + 1) - T(0.5)
        for round in evaluator.rounds
            @threads for (res_index, dst_begin, dst_end, _, _) in round
                j = dst_begin - 1
                n = dst_end - j
                @inbounds res[res_index] = inv_symmetry[res_index] * (
                    dot_inplace(n, u, j, x, num_vars - n) -
                    inv_density[res_index])
            end
        end
    else
        let
            output_index = output_indices[1]
            if output_index > 0
                first = -one(T)
                b_idx = num_vars - num_stages + 1
                @simd for i = b_idx : num_vars
                    @inbounds first += x[i]
                end
                @inbounds res[output_index] = first
            end
        end
        let
            output_index = output_indices[2]
            if output_index > 0
                b_idx = num_vars - num_stages + 1
                @inbounds res[output_index] =
                    dot_inplace(num_stages - 1, u, 0, x, b_idx) - T(0.5)
            end
        end
        for round in evaluator.rounds
            @threads for (res_index, dst_begin, dst_end, _, _) in round
                output_index = output_indices[res_index]
                if output_index > 0
                    j = dst_begin - 1
                    n = dst_end - j
                    @inbounds res[output_index] = inv_symmetry[res_index] * (
                        dot_inplace(n, u, j, x, num_vars - n) -
                        inv_density[res_index])
                end
            end
        end
    end
end

################################################################################

function evaluate_jacobian!(jac::Matrix{T}, x::Vector{T},
        evaluator::RKOCEvaluator{T})::Nothing where {T <: Real}
    @assert(length(x) == evaluator.num_vars)
    u = evaluator.u
    num_stages = evaluator.num_stages
    num_vars = evaluator.num_vars
    output_indices = evaluator.output_indices
    populate_u!(evaluator, x)
    @threads for var_idx = 1 : num_vars
        @inbounds v = evaluator.vs[threadid()]
        populate_v!(evaluator, v, x, var_idx - 1)
        if length(output_indices) == 0
            @inbounds jac[1, var_idx] = T(var_idx + num_stages > num_vars)
            let
                n = num_stages - 1
                m = num_vars - n
                result = dot_inplace(n, v, 0, x, m)
                if var_idx + n > num_vars
                    @inbounds result += u[var_idx - m]
                end
                @inbounds jac[2, var_idx] = result
            end
            for round in evaluator.rounds
                for (res_index, dst_begin, dst_end, _, _) in round
                    n = dst_end - dst_begin + 1
                    m = num_vars - n
                    result = dot_inplace(n, v, dst_begin - 1, x, m)
                    if var_idx + n > num_vars
                        @inbounds result += u[dst_begin - 1 + var_idx - m]
                    end
                    @inbounds jac[res_index, var_idx] = result
                end
            end
        else
            let
                output_index = output_indices[1]
                if output_index > 0
                    @inbounds jac[output_index, var_idx] =
                        T(var_idx + num_stages > num_vars)
                end
            end
            let
                output_index = output_indices[2]
                if output_index > 0
                    n = num_stages - 1
                    m = num_vars - n
                    result = dot_inplace(n, v, 0, x, m)
                    if var_idx + n > num_vars
                        @inbounds result += u[var_idx - m]
                    end
                    @inbounds jac[output_index, var_idx] = result
                end
            end
            for round in evaluator.rounds
                for (res_index, dst_begin, dst_end, _, _) in round
                    output_index = output_indices[res_index]
                    if output_index > 0
                        n = dst_end - dst_begin + 1
                        m = num_vars - n
                        result = dot_inplace(n, v, dst_begin - 1, x, m)
                        if var_idx + n > num_vars
                            @inbounds result += u[dst_begin - 1 + var_idx - m]
                        end
                        @inbounds jac[output_index, var_idx] = result
                    end
                end
            end
        end
    end
end

function evaluate_error_jacobian!(jac::Matrix{T}, x::Vector{T},
        evaluator::RKOCEvaluator{T})::Nothing where {T <: Real}
    @assert(length(x) == evaluator.num_vars)
    u = evaluator.u
    num_stages = evaluator.num_stages
    num_vars = evaluator.num_vars
    output_indices = evaluator.output_indices
    inv_symmetry = evaluator.inv_symmetry
    populate_u!(evaluator, x)
    @threads for var_idx = 1 : num_vars
        @inbounds v = evaluator.vs[threadid()]
        populate_v!(evaluator, v, x, var_idx - 1)
        if length(output_indices) == 0
            @inbounds jac[1, var_idx] = T(var_idx + num_stages > num_vars)
            let
                n = num_stages - 1
                m = num_vars - n
                result = dot_inplace(n, v, 0, x, m)
                if var_idx + n > num_vars
                    @inbounds result += u[var_idx - m]
                end
                @inbounds jac[2, var_idx] = result
            end
            for round in evaluator.rounds
                for (res_index, dst_begin, dst_end, _, _) in round
                    n = dst_end - dst_begin + 1
                    m = num_vars - n
                    result = dot_inplace(n, v, dst_begin - 1, x, m)
                    if var_idx + n > num_vars
                        @inbounds result += u[dst_begin - 1 + var_idx - m]
                    end
                    result *= inv_symmetry[res_index]
                    @inbounds jac[res_index, var_idx] = result
                end
            end
        else
            let
                output_index = output_indices[1]
                if output_index > 0
                    @inbounds jac[output_index, var_idx] =
                        T(var_idx + num_stages > num_vars)
                end
            end
            let
                output_index = output_indices[2]
                if output_index > 0
                    n = num_stages - 1
                    m = num_vars - n
                    result = dot_inplace(n, v, 0, x, m)
                    if var_idx + n > num_vars
                        @inbounds result += u[var_idx - m]
                    end
                    @inbounds jac[output_index, var_idx] = result
                end
            end
            for round in evaluator.rounds
                for (res_index, dst_begin, dst_end, _, _) in round
                    output_index = output_indices[res_index]
                    if output_index > 0
                        n = dst_end - dst_begin + 1
                        m = num_vars - n
                        result = dot_inplace(n, v, dst_begin - 1, x, m)
                        if var_idx + n > num_vars
                            @inbounds result += u[dst_begin - 1 + var_idx - m]
                        end
                        result *= inv_symmetry[res_index]
                        @inbounds jac[output_index, var_idx] = result
                    end
                end
            end
        end
    end
end

################################################################################

import Base: adjoint

struct RKOCEvaluatorAdjointProxy{T <: Real}
    evaluator::RKOCEvaluator{T}
end

function adjoint(evaluator::RKOCEvaluator{T}) where {T <: Real}
    RKOCEvaluatorAdjointProxy{T}(evaluator)
end

function (evaluator::RKOCEvaluator{T})(x::Vector{T}) where {T <: Real}
    residual = Vector{T}(undef, evaluator.num_constrs)
    evaluate_error_coefficients!(residual, x, evaluator)
    residual
end

function (proxy::RKOCEvaluatorAdjointProxy{T})(x::Vector{T}) where {T <: Real}
    jacobian = Matrix{T}(undef,
                         proxy.evaluator.num_constrs, proxy.evaluator.num_vars)
    evaluate_error_jacobian!(jacobian, x, proxy.evaluator)
    jacobian
end

function constrain!(x::Vector{T},
                    evaluator::RKOCEvaluator{T})::T where {T <: Real}
    num_vars, num_constrs = evaluator.num_vars, evaluator.num_constrs
    x_new = Vector{T}(undef, num_vars)
    residual = Vector{T}(undef, num_constrs)
    jacobian = Matrix{T}(undef, num_constrs, num_vars)
    direction = Vector{T}(undef, num_vars)
    evaluate_error_coefficients!(residual, x, evaluator)
    obj_old = norm2(residual)
    while true
        evaluate_error_jacobian!(jacobian, x, evaluator)
        ldiv!(direction, qrfactUnblocked!(jacobian), residual)
        @simd ivdep for i = 1 : num_vars
            @inbounds x_new[i] = x[i] - direction[i]
        end
        evaluate_error_coefficients!(residual, x_new, evaluator)
        obj_new = norm2(residual)
        if obj_new < obj_old
            @simd ivdep for i = 1 : num_vars
                @inbounds x[i] = x_new[i]
            end
            obj_old = obj_new
        else
            return sqrt(obj_old)
        end
    end
end

function compute_order!(x::Vector{T}, threshold::T; verbose::Bool=false,
                        prefix::String="")::Int where {T <: Real}
    num_stages = compute_stages(x)
    order = 2
    while true
        rmk(prefix, "Deriving conditions for order ", order, "...";
            verbose=verbose)
        evaluator = RKOCEvaluator{T}(order, num_stages)
        rmk(prefix, "Testing constraints for order ", order, "...";
            verbose=verbose)
        obj_new = constrain!(x, evaluator)
        say(prefix, "Residual for order ", lpad(string(order), 2, ' '),
            ": ", obj_new; verbose=verbose)
        if obj_new <= threshold
            order += 1
        else
            return order - 1
        end
    end
end

function compute_stages(x::Vector{T})::Int where {T <: Real}
    num_stages = div(isqrt(8 * length(x) + 1) - 1, 2)
    @assert(length(x) == div(num_stages * (num_stages + 1), 2))
    num_stages
end

################################################################################

# This is the classical Runge-Kutta method, for which no introduction should
# be needed. (If you don't know what this is, you probably need to do some more
# background reading before using this package!)
rk4_table(::Type{T}) where {T <: Real} = T[
    inv(T(2)), T(0), inv(T(2)), T(0), T(0), T(1),
    inv(T(6)), inv(T(3)), inv(T(3)), inv(T(6))]

# This is the Runge-Kutta method obtained by applying Richardson extrapolation
# to n independent executions of Euler's method using step sizes h, h/2, h/3,
# ..., h/n, yielding a method of order n. The methods obtained in this fashion
# are not practically useful, but provide a simple proof that Runge-Kutta
# methods exist of any order using a quadratic number of stages.
function extrapolated_euler_table(order::Int)::Vector{Rational{BigInt}}
    result = Vector{Rational{BigInt}}[]
    skip = 0
    for i = 2 : order
        for j = 1 : i - 1
            push!(result, vcat(
                [Rational{BigInt}(1, i)],
                zeros(Rational{BigInt}, skip),
                [Rational{BigInt}(1, i) for _ = 1 : j - 1]))
        end
        skip += i - 1
    end
    num_stages = div(order * (order - 1), 2) + 1
    mat = zeros(Rational{BigInt}, order, num_stages)
    skip = 0
    for i = 1 : order
        mat[i, 1] = Rational{BigInt}(1, i)
        for j = skip + 2 : skip + i
            mat[i, j] = Rational{BigInt}(1, i)
        end
        skip += i - 1
    end
    lhs = [Rational{BigInt}(1, j)^i for i = 0 : order - 1, j = 1 : order]
    rhs = [Rational{BigInt}(i == 1, 1) for i = 1 : order]
    push!(result, transpose(mat) * (lhs \ rhs))
    vcat(result...)
end

function extrapolated_euler_table(::Type{T},
        order::Int)::Vector{T} where {T <: Real}
    T.(extrapolated_euler_table(order))
end

# This is the 5th-order method presented on p. 206 of the following paper.
# Interestingly, it is presented not as a standalone method or in an embedded
# pair, but as the highest-order component of an embedded quintuple of orders
# 5(4,3,2,1). Cash and Karp themselves named it "RKFNC"; I'm not sure why.
# Numerical Recipes calls this method "RKCK."
# Cash and Karp 1990, "A Variable Order Runge-Kutta Method for Initial Value
#                      Problems with Rapidly Varying Right-Hand Sides"
rkck5_table(::Type{T}) where {T <: Real} = T[
    inv(T(5)), T(3)/T(40), T(9)/T(40), T(3)/T(10), T(-9)/T(10), T(6)/T(5),
    T(-11)/T(54), T(5)/T(2), T(-70)/T(27), T(35)/T(27), T(1631)/T(55296),
    T(175)/T(512), T(575)/T(13824), T(44275)/T(110592), T(253)/T(4096),
    T(37)/T(378), T(0), T(250)/T(621), T(125)/T(594), T(0), T(512)/T(1771)]

# This is the higher-order component of the 5(4) embedded pair presented on
# p. 23 of the following paper. Dormand and Prince call this method "RK5(4)7M,"
# but it has become commonly known as "DOPRI5" following a popular Fortran
# implementation.
# Dormand and Prince 1980, "A family of embedded Runge-Kutta formulae"
dopri5_table(::Type{T}) where {T <: Real} = T[
    inv(T(5)), T(3)/T(40), T(9)/T(40), T(44)/T(45), T(-56)/T(15), T(32)/T(9),
    T(19372)/T(6561), T(-25360)/T(2187), T(64448)/T(6561), T(-212)/T(729),
    T(9017)/T(3168), T(-355)/T(33), T(46732)/T(5247), T(49)/T(176),
    T(-5103)/T(18656), T(35)/T(384), T(0), T(500)/T(1113), T(125)/T(192),
    T(-2187)/T(6784), T(11)/T(84), T(35)/T(384), T(0), T(500)/T(1113),
    T(125)/T(192), T(-2187)/T(6784), T(11)/T(84), T(0)]

# This is the 8th-order method presented on p. 65 of Fehlberg's 1968 NASA
# paper as the higher-order component of the embedded pair RK7(8).
# Fehlberg 1968, "Classical Fifth-, Sixth-, Seventh-, and Eighth-Order
#                 Runge-Kutta Formulas with Stepsize Control"
rkf8_table(::Type{T}) where {T <: Real} = T[
    T(2)/T(27), inv(T(36)), inv(T(12)), inv(T(24)), T(0), inv(T(8)),
    T(5)/T(12), T(0), T(-25)/T(16), T(25)/T(16), inv(T(20)), T(0), T(0),
    inv(T(4)), inv(T(5)), T(-25)/T(108), T(0), T(0), T(125)/T(108),
    T(-65)/T(27), T(125)/T(54), T(31)/T(300), T(0), T(0), T(0), T(61)/T(225),
    T(-2)/T(9), T(13)/T(900), T(2)/T(1), T(0), T(0), T(-53)/T(6), T(704)/T(45),
    T(-107)/T(9), T(67)/T(90), T(3)/T(1), T(-91)/T(108), T(0), T(0),
    T(23)/T(108), T(-976)/T(135), T(311)/T(54), T(-19)/T(60), T(17)/T(6),
    T(-1)/T(12), T(2383)/T(4100), T(0), T(0), T(-341)/T(164), T(4496)/T(1025),
    T(-301)/T(82), T(2133)/T(4100), T(45)/T(82), T(45)/T(164), T(18)/T(41),
    T(3)/T(205), T(0), T(0), T(0), T(0), T(-6)/T(41), T(-3)/T(205),
    T(-3)/T(41), T(3)/T(41), T(6)/T(41), T(0), T(-1777)/T(4100), T(0), T(0),
    T(-341)/T(164), T(4496)/T(1025), T(-289)/T(82), T(2193)/T(4100),
    T(51)/T(82), T(33)/T(164), T(12)/T(41), T(0), T(1), T(0), T(0), T(0), T(0),
    T(0), T(34)/T(105), T(9)/T(35), T(9)/T(35), T(9)/T(280), T(9)/T(280), T(0),
    T(41)/T(840), T(41)/T(840)]

################################################################################

struct RKSolver{T <: Real}
    num_stages::Int
    coeffs::Vector{T}
    dimension::Int
    y_temp::Vector{T}
    k_temp::Matrix{T}
end

function RKSolver{T}(coeffs::Vector{T}, dimension::Int) where {T <: Real}
    num_stages = compute_stages(coeffs)
    RKSolver{T}(num_stages, coeffs, dimension, Vector{T}(undef, dimension),
        Matrix{T}(undef, dimension, num_stages))
end

function runge_kutta_step!(f!, y::Vector{T}, step_size::T,
        solver::RKSolver{T}) where {T <: Real}
    s, x, dim, y_temp, k = solver.num_stages, solver.coeffs,
        solver.dimension, solver.y_temp, solver.k_temp
    @inbounds f!(view(k, :, 1), y)
    @simd ivdep for d = 1 : dim
        @inbounds k[d, 1] *= step_size
    end
    n = 1
    for i = 2 : s
        @simd ivdep for d = 1 : dim
            @inbounds y_temp[d] = y[d]
        end
        for j = 1 : i - 1
            @simd ivdep for d = 1 : dim
                @inbounds y_temp[d] += x[n] * k[d, j]
            end
            n += 1
        end
        @inbounds f!(view(k, :, i), y_temp)
        @simd ivdep for d = 1 : dim
            @inbounds k[d, i] *= step_size
        end
    end
    for i = 1 : s
        @simd ivdep for d = 1 : dim
            @inbounds y[d] += x[n] * k[d, i]
        end
        n += 1
    end
end

################################################################################

function compute_butcher_weights!(m::Matrix{T}, A::Matrix{T},
        dependencies::Vector{Vector{Int}}) where {T <: Number}
    num_stages, num_constrs = size(m, 1), size(m, 2)
    @inbounds for i = 1 : num_constrs
        dep = dependencies[i]
        n = length(dep)
        if n == 0
            @simd ivdep for j = 1 : num_stages
                m[j,i] = one(T)
            end
        elseif n == 1
            d = dep[1]
            for j = 1 : num_stages
                m[j,i] = zero(T)
                @simd for k = 1 : num_stages
                    m[j,i] += A[j,k] * m[k,d]
                end
            end
        else
            d, e = dep[1], dep[2]
            @simd ivdep for j = 1 : num_stages
                m[j,i] = m[j,d] * m[j,e]
            end
        end
    end
end

function backprop_butcher_weights!(u::Matrix{T}, A::Matrix{T}, b::Vector{T},
        m::Matrix{T}, p::Vector{T}, children::Vector{Int},
        siblings::Vector{Vector{Tuple{Int,Int}}}) where {T <: Number}
    num_stages, num_constrs = size(u, 1), size(u, 2)
    @inbounds for r = 0 : num_constrs - 1
        i = num_constrs - r
        x = p[i]
        @simd ivdep for j = 1 : num_stages
            u[j,i] = x * b[j]
        end
        c = children[i]
        if c > 0
            for j = 1 : num_stages
                @simd for k = 1 : num_stages
                    u[j,i] += A[k,j] * u[k,c]
                end
            end
        end
        for (s, t) in siblings[i]
            @simd ivdep for j = 1 : num_stages
                u[j,i] += m[j,s] .* u[j,t]
            end
        end
    end
end

function find_children_siblings(dependencies::Vector{Vector{Int}})
    children = [0 for _ in dependencies]
    siblings = [Tuple{Int,Int}[] for _ in dependencies]
    for (i, dep) in enumerate(dependencies)
        if length(dep) == 1
            children[dep[1]] = i
        elseif length(dep) == 2
            push!(siblings[dep[1]], (dep[2], i))
            push!(siblings[dep[2]], (dep[1], i))
        end
    end
    children, siblings
end

################################################################################

struct RKOCBackpropEvaluator{T <: Real}
    order::Int
    num_stages::Int
    num_constrs::Int
    dependencies::Vector{Vector{Int}}
    children::Vector{Int}
    siblings::Vector{Vector{Tuple{Int,Int}}}
    inv_density::Vector{T}
    m::Matrix{T} # Matrix of Butcher weights
    u::Matrix{T} # Gradients of Butcher weights
    p::Vector{T} # Vector of doubled residuals
    q::Vector{T} # Dot products of Butcher weights
end

function RKOCBackpropEvaluator{T}(order::Int, num_stages::Int) where {T <: Real}
    trees = rooted_trees(order)
    num_constrs = sum(length.(trees))
    dependencies = dependency_table(trees)
    children, siblings = find_children_siblings(dependencies)
    inv_density = inv.(T.(butcher_density.(vcat(trees...))))
    RKOCBackpropEvaluator(order, num_stages, num_constrs,
        dependencies, children, siblings, inv_density,
        Matrix{T}(undef, num_stages, num_constrs),
        Matrix{T}(undef, num_stages, num_constrs),
        Vector{T}(undef, num_constrs),
        Vector{T}(undef, num_constrs))
end

function evaluate_residual2(A::Matrix{T}, b::Vector{T},
        evaluator::RKOCBackpropEvaluator{T}) where {T <: Real}
    num_constrs, inv_density = evaluator.num_constrs, evaluator.inv_density
    m, q = evaluator.m, evaluator.q
    compute_butcher_weights!(m, A, evaluator.dependencies)
    mul!(q, transpose(m), b)
    residual = zero(T)
    @inbounds @simd ivdep for j = 1 : num_constrs
        x = q[j] - inv_density[j]
        residual += x * x
    end
    residual
end

function evaluate_gradient!(gA::Matrix{T}, gb::Vector{T}, A::Matrix{T},
        b::Vector{T}, evaluator::RKOCBackpropEvaluator{T}) where {T <: Real}
    num_stages, num_constrs = evaluator.num_stages, evaluator.num_constrs
    inv_density, children = evaluator.inv_density, evaluator.children
    m, u, p, q = evaluator.m, evaluator.u, evaluator.p, evaluator.q
    compute_butcher_weights!(m, A, evaluator.dependencies)
    mul!(q, transpose(m), b)
    residual = zero(T)
    @inbounds @simd ivdep for j = 1 : num_constrs
        x = q[j] - inv_density[j]
        residual += x * x
        p[j] = dbl(x)
    end
    backprop_butcher_weights!(u, A, b, m, p, children, evaluator.siblings)
    @inbounds for t = 1 : num_stages
        @simd ivdep for s = 1 : num_stages
            gA[s,t] = zero(T)
        end
    end
    @inbounds for i = 1 : num_constrs
        j = children[i]
        if j > 0
            for t = 1 : num_stages
                @simd ivdep for s = 1 : num_stages
                    gA[s,t] += u[s,j] * m[t,i]
                end
            end
        end
    end
    mul!(gb, m, p)
    residual
end

################################################################################

function populate_explicit!(A::Matrix{T}, b::Vector{T}, x::Vector{T},
        n::Int) where {T <: Number}
    k = 0
    for i = 1 : n
        @simd ivdep for j = 1 : i - 1
            @inbounds A[i,j] = x[k + j]
        end
        k += i - 1
        @simd ivdep for j = i : n
            @inbounds A[i,j] = zero(T)
        end
    end
    @simd ivdep for i = 1 : n
        @inbounds b[i] = x[k + i]
    end
end

function populate_explicit!(x::Vector{T}, A::Matrix{T}, b::Vector{T},
        n::Int) where {T <: Number}
    k = 0
    for i = 2 : n
        @simd ivdep for j = 1 : i - 1
            @inbounds x[k + j] = A[i,j]
        end
        k += i - 1
    end
    @simd ivdep for i = 1 : n
        @inbounds x[k + i] = b[i]
    end
end

################################################################################

# struct RKOCBackpropFSGDOptimizer{T <: Real}
#     evaluator::RKOCBackpropEvaluator{T}
#     A::Matrix{T}
#     b::Vector{T}
#     x::Vector{T}
#     gA::Matrix{T}
#     gb::Vector{T}
#     gx::Vector{T}
# end

# function RKOCBackpropFSGDOptimizer{T}(order::Int, num_stages::Int,
#         x_init::Vector{S}) where {S <: Real, T <: Real}
#     evaluator = RKOCBackpropEvaluator{T}(order, num_stages)
#     num_vars = div(num_stages * (num_stages + 1), 2)
#     @assert length(x_init) == num_vars
#     RKOCBackpropFSGDOptimizer{T}(evaluator,
#         Matrix{T}(undef, num_stages, num_stages),
#         Vector{T}(undef, num_stages),
#         T.(x_init),
#         Matrix{T}(undef, num_stages, num_stages),
#         Vector{T}(undef, num_stages),
#         Vector{T}(undef, num_vars))
# end

# function step!(optimizer::RKOCBackpropFSGDOptimizer{T},
#         step_size::T, num_steps::Int) where {T <: Real}
#     evaluator = optimizer.evaluator
#     num_stages = evaluator.num_stages
#     num_vars = div(num_stages * (num_stages + 1), 2)
#     A, b, x = optimizer.A, optimizer.b, optimizer.x
#     gA, gb, gx = optimizer.gA, optimizer.gb, optimizer.gx
#     for _ = 1 : num_steps
#         populate_explicit!(A, b, x, num_stages)
#         evaluate_gradient!(gA, gb, A, b, evaluator)
#         populate_explicit!(gx, gA, gb, num_stages)
#         normalize!(gx)
#         @simd ivdep for i = 1 : num_vars
#             @inbounds x[i] -= step_size * gx[i]
#         end
#     end
#     evaluate_residual2(A, b, evaluator)
# end

end # module RKTK2
