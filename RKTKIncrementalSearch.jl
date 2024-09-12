using DZOptimization
using DZOptimization.PCG
using GenericLinearAlgebra
using LinearAlgebra
using MultiFloats
using Printf
using RungeKuttaToolKit
using RungeKuttaToolKit.RKCost
using RungeKuttaToolKit.RKParameterization


function construct_optimizer(
    prob::RKOCOptimizationProblem{T},
    x::AbstractVector{T},
) where {T}
    n = length(x)
    @assert n == prob.param.num_variables
    return LBFGSOptimizer(prob, prob', QuadraticLineSearch(),
        x, sqrt(n * eps(T)), n)
end


function autosearch!(
    x::AbstractVector{T},
    param::AbstractRKParameterization{T},
    epsilon::T,
    radius::T,
    all_trees::AbstractVector{LevelSequence};
    verbose::Bool=false,
) where {T}
    excluded_trees = Int[]
    A = Matrix{T}(undef, param.num_stages, param.num_stages)
    dA = Matrix{T}(undef, param.num_stages, param.num_stages)
    b = Vector{T}(undef, param.num_stages)
    db = Vector{T}(undef, param.num_stages)
    for i = 1:length(all_trees)
        trees = deleteat!(all_trees[1:i], excluded_trees)
        ev = RKOCEvaluator{T}(trees, param.num_stages)
        prob = RKOCOptimizationProblem(ev,
            RKCostL2{T}(), param, A, dA, b, db)
        opt = construct_optimizer(prob, x)
        start = time_ns()
        while !opt.has_terminated[]
            step!(opt)
            if !(maximum(abs, opt.current_point) < radius)
                break
            end
            if time_ns() - start > 6.0e11
                println("WARNING: Optimization timed out.")
                break
            end
        end
        param(A, b, opt.current_point)
        max_residual = ev(RKCostLInfinity{T}(), A, b)
        if max_residual <= epsilon
            if verbose
                println("Accepted tree ", i,
                    " with max residual ", Float64(max_residual))
                flush(stdout)
            end
            copy!(x, opt.current_point)
            if length(trees) >= param.num_variables
                _, s, _ = svd!(ev'(param, x))
                if length(s) >= param.num_variables
                    sigma = minimum(abs, s)
                    if minimum(abs, s) > epsilon
                        return (trees, sigma)
                    end
                end
            end
        else
            if verbose
                println("Rejected tree ", i,
                    " with max residual ", Float64(max_residual))
                flush(stdout)
            end
            push!(excluded_trees, i)
        end
    end
    return (deleteat!(copy(all_trees), excluded_trees), zero(T))
end


function main(
    param::AbstractRKParameterization{T},
    param_big::AbstractRKParameterization{BigFloat},
    epsilon::T,
    radius::T,
    all_trees::AbstractVector{LevelSequence},
) where {T}
    for seed = typemin(UInt64):typemax(UInt64)
        println("Seed: ", seed)
        flush(stdout)
        x = random_array(seed, T, param.num_variables)
        trees, sigma = autosearch!(x, param, epsilon, radius, all_trees;
            verbose=true)
        println("Discovered method determined by ",
            length(trees), " order conditions (", sigma, ").")
        flush(stdout)

        ev_big = RKOCEvaluator{BigFloat}(trees, param_big.num_stages)
        prob_big = RKOCOptimizationProblem(
            ev_big, RKCostL2{BigFloat}(), param_big)
        opt_big = construct_optimizer(prob_big, BigFloat.(x))
        while !opt_big.has_terminated[]
            step!(opt_big)
        end
        _, s, _ = svd!(ev_big'(param_big, opt_big.current_point))
        sigma_big = minimum(abs, s)
        println("Sigma: ", sigma_big)
        println('{')
        for (i, c) in pairs(opt_big.current_point)
            if i == lastindex(opt_big.current_point)
                @printf("    %+.100f\n", c)
            else
                @printf("    %+.100f,\n", c)
            end
        end
        println('}')
        flush(stdout)
    end
end


setprecision(BigFloat, 512)
main(
    RKParameterizationExplicit{Float64x2}(parse(Int, ARGS[1])),
    RKParameterizationExplicit{BigFloat}(parse(Int, ARGS[1])),
    Float64x2(1.0e-25), Float64x2(100.0),
    all_rooted_trees(12; tree_ordering=:butcher),
)
