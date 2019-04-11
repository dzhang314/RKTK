module RKTK2

export RKOCEvaluator, evaluate_residual!, evaluate_jacobian!,
    evaluate_error_coefficients!, evaluate_error_jacobian!

import Base: adjoint

using Base.Threads: @threads, nthreads, threadid

using DZMisc: dbl, RootedTree, rooted_tree_count, rooted_trees,
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
    total_tree_count = rooted_tree_count(order)
    while total_tree_count < max_index
        total_tree_count += rooted_tree_count(order += 1)
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
    # TODO: These assertions should not be necessary.
    @assert(indices[1] == 1)
    @assert(indices[2] == 2)
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
    if length(output_indices) == 0
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
    if length(output_indices) == 0
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
        if length(output_indices) == 0
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
        if length(output_indices) == 0
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

struct RKOCEvaluatorAdjointProxy{T <: Real}
    evaluator::RKOCEvaluator{T}
end

function adjoint(evaluator::RKOCEvaluator{T}) where {T <: Real}
    RKOCEvaluatorAdjointProxy{T}(evaluator)
end

function (evaluator::RKOCEvaluator{T})(x::Vector{T}) where {T <: Real}
    residual = Vector{T}(undef, evaluator.num_constrs)
    evaluate_residual!(residual, x, evaluator)
    residual
end

function (proxy::RKOCEvaluatorAdjointProxy{T})(x::Vector{T}) where {T <: Real}
    jacobian = Matrix{T}(undef,
                         proxy.evaluator.num_constrs, proxy.evaluator.num_vars)
    evaluate_jacobian!(jacobian, x, proxy.evaluator)
    jacobian
end

end # module RKTK2
