using DZOptimization
using DZOptimization.PCG
using GenericLinearAlgebra
using LinearAlgebra
using MultiFloats
using Printf
using RungeKuttaToolKit
using RungeKuttaToolKit.RKCost
using RungeKuttaToolKit.RKParameterization


push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using RKTKUtilities


function autosearch!(
    active_trees::AbstractVector{LevelSequence},
    x::AbstractVector{T},
    param::AbstractRKParameterization{T},
    param_big::AbstractRKParameterization{BigFloat},
    order::Int,
    height_limit::Int,
    epsilon::T,
    radius::T,
) where {T}

    @assert param.num_stages == param_big.num_stages
    @assert param.num_variables == param_big.num_variables
    @assert length(x) == param.num_variables

    trees = [tree for tree in rooted_trees(order)
             if maximum(tree.data) <= height_limit]
    while !isempty(trees)

        res_ev = RKOCEvaluator{T}(trees, param.num_stages)
        _, index = findmin(abs, res_ev(param(x)...))
        println("Attempting to add tree: ", trees[index])

        trial_trees = push!(copy(active_trees), trees[index])
        trial_ev = RKOCEvaluator{T}(trial_trees, param.num_stages)
        trial_opt = construct_optimizer(RKOCOptimizationProblem(
                trial_ev, RKCostL2{T}(), param), x)

        println(compute_table_row(trial_opt))
        while !trial_opt.has_terminated[]
            step!(trial_opt)
            if !(maximum(abs, trial_opt.current_point) < radius)
                break
            end
            if trial_opt.iteration_count[] % 5000 == 0
                println(compute_table_row(trial_opt))
            end
        end
        println(compute_table_row(trial_opt))

        max_residual = trial_ev(RKCostLInfinity{T}(),
            param(trial_opt.current_point)...)
        if max_residual < epsilon
            @printf("Succeeded with max residual: %.6e\n", max_residual)
            push!(active_trees, trees[index])
            deleteat!(trees, index)

            ev_big = RKOCEvaluator{BigFloat}(
                active_trees, param_big.num_stages)
            prob_big = RKOCOptimizationProblem(
                ev_big, RKCostL2{BigFloat}(), param_big)
            opt_big = construct_optimizer(
                prob_big, BigFloat.(trial_opt.current_point))
            while !opt_big.has_terminated[]
                step!(opt_big)
            end

            max_residual = ev_big(RKCostLInfinity{BigFloat}(),
                param_big(opt_big.current_point)...)
            @printf("Refined max residual: %.6e\n", max_residual)
            @assert max_residual < 1.0e-100
            copy!(x, T.(opt_big.current_point))

            _, s, _ = svd!(ev_big'(param_big, opt_big.current_point))
            i = lastindex(s)
            while i >= 1 && s[i] < 1.0e-100
                i -= 1
            end
            rank = i
            if i == lastindex(s)
                @printf("Estimated Jacobian rank: %d (singular value %.6e)\n",
                    rank, s[i])
            else
                @printf("Estimated Jacobian rank: %d (spectral gap %.6e)\n",
                    rank, s[i] / s[i+1])
            end
            @printf("Constrained %d of %d variables.\n",
                rank, param.num_variables)

            if rank >= param.num_variables
                return true
            end

        else
            println("Failed with max residual: ", max_residual)
            deleteat!(trees, index)
        end

        flush(stdout)
    end

    return false
end


function main(
    param::AbstractRKParameterization{T},
    param_big::AbstractRKParameterization{BigFloat},
    height_limit::Int,
    epsilon::T,
    radius::T,
    min_seed::UInt64,
) where {T}

    for seed = min_seed:typemax(UInt64)

        println("Seed: ", seed)
        flush(stdout)
        x = random_array(seed, T, param.num_variables)
        active_trees = LevelSequence[]

        for order = 1:99
            reached_full_rank = autosearch!(
                active_trees, x, param, param_big,
                order, height_limit, epsilon, radius)

            ev_big = RKOCEvaluator{BigFloat}(
                active_trees, param_big.num_stages)
            prob_big = RKOCOptimizationProblem(
                ev_big, RKCostL2{BigFloat}(), param_big)
            opt_big = construct_optimizer(prob_big, BigFloat.(x))
            while !opt_big.has_terminated[]
                step!(opt_big)
            end
            println('{')
            for (i, c) in pairs(opt_big.current_point)
                if i == lastindex(opt_big.current_point)
                    @printf("    %+.100f\n", c)
                else
                    @printf("    %+.100f,\n", c)
                end
            end
            println('}')
            copy!(x, T.(opt_big.current_point))

            trees = rooted_trees(order)
            count = sum(tree in active_trees for tree in trees)
            @printf("Method satisfies %d of %d conditions for order %d.\n",
                count, length(trees), order)
            if reached_full_rank
                break
            end
        end

    end

    return nothing
end


setprecision(BigFloat, 512)
main(
    RKParameterizationParallelExplicit{Float64x2}(parse(Int, ARGS[1]) - 1, parse(Int, ARGS[2])),
    RKParameterizationParallelExplicit{BigFloat}(parse(Int, ARGS[1]) - 1, parse(Int, ARGS[2])),
    parse(Int, ARGS[1]), Float64x2(1.0e-20), Float64x2(100.0),
    parse(UInt64, ARGS[3]),
)
