using Base.Threads
using DZOptimization
using DZOptimization.PCG
using MultiFloats
using Printf
using RungeKuttaToolKit
using RungeKuttaToolKit.RKCost
using RungeKuttaToolKit.RKParameterization


push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using RKTKUtilities


const WRITE_TERM = ((stdout isa Base.TTY || stdout isa Base.PipeEndpoint) &&
                    (nthreads() == 1))
const WRITE_FILE = !get_flag!(["no-file"])


function fprintln(io::IO, args...)
    @static if WRITE_TERM
        println(stdout, args...)
        flush(stdout)
    end
    @static if WRITE_FILE
        println(io, args...)
        flush(io)
    end
    return nothing
end


function fprint_error_coefficents(
    io::IO,
    param::AbstractRKParameterization{BigFloat},
    x::AbstractVector{BigFloat},
)
    k = 0
    for order = 1:99
        trees = rooted_trees(order; tree_ordering=:butcher)
        ev = RKOCEvaluator{BigFloat}(trees, param.num_stages)
        residuals = ev(param(x)...)
        count = 0
        for (tree, residual) in zip(trees, residuals)
            if abs(residual) < 1.0e-100
                count += 1
            end
            fprintln(io, @sprintf("t_%d ", k += 1), tree.data, ": ",
                @sprintf("%+.100f", residual / butcher_symmetry(tree)))
        end
        if count == length(trees)
            fprintln(io, @sprintf(
                "Satisfied all %d conditions for order %d.", count, order))
        else
            fprintln(io, @sprintf(
                "Satisfied %d of %d conditions for order %d.",
                count, length(trees), order))
        end
        fprintln(io)
        if iszero(count)
            break
        end
    end
end


function autosearch!(
    io::IO,
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

    active_set = Set{LevelSequence}(active_trees)
    trees = [tree for tree in rooted_trees(order)
             if (maximum(tree.data) <= height_limit) && !(tree in active_set)]
    while !isempty(trees)

        res_ev = RKOCEvaluator{T}(trees, param.num_stages)
        _, index = findmin(abs, res_ev(param(x)...))
        fprintln(io, "Attempting to add rooted tree: ", trees[index].data)

        trial_trees = push!(copy(active_trees), trees[index])
        trial_ev = RKOCEvaluator{T}(trial_trees, param.num_stages)
        trial_prob = RKOCOptimizationProblem(trial_ev, RKCostL2{T}(), param)
        trial_opt = construct_optimizer(trial_prob, x)

        fprintln(io, TABLE_HEADER)
        fprintln(io, TABLE_SEPARATOR)
        fprintln(io, compute_table_row(trial_opt))
        while !trial_opt.has_terminated[]
            step!(trial_opt)
            if !(maximum(abs, trial_opt.current_point) < radius)
                break
            end
            if trial_opt.iteration_count[] >= 10^6
                break
            end
            if (reset_occurred(trial_opt) ||
                (trial_opt.iteration_count[] % 1000 == 0))
                fprintln(io, compute_table_row(trial_opt))
            end
        end
        fprintln(io, compute_table_row(trial_opt))

        param(trial_prob.A, trial_prob.b, trial_opt.current_point)
        max_residual = trial_ev(RKCostLInfinity{T}(),
            trial_prob.A, trial_prob.b)

        if max_residual < epsilon
            fprintln(io, "Sucessfully added rooted tree: ", trees[index].data)
            fprintln(io, @sprintf("Pre-refinement residual: %.6e",
                max_residual))
            push!(active_trees, trees[index])
            deleteat!(trees, index)

            x_big, max_residual, sigma, iter_count = refine_bigfloat(
                active_trees, param_big, trial_opt.current_point)
            @assert max_residual < 1.0e-100
            if iszero(max_residual)
                fprintln(io, "Successfully refined to exact accuracy.")
            else
                fprintln(io, @sprintf(
                    "Successfully refined to %d-digit accuracy.",
                    floor(Int, -log10(max_residual))))
            end
            fprintln(io, @sprintf("Refinement iteration count: %d",
                iter_count))
            copy!(x, T.(x_big))

            i = lastindex(sigma)
            while i >= 1 && sigma[i] < 1.0e-100
                i -= 1
            end
            rank = i
            if i == lastindex(sigma)
                fprintln(io, @sprintf("Estimated Jacobian rank: %d", rank))
                fprintln(io, @sprintf("Singular value: %.100f", sigma[i]))
            else
                fprintln(io, @sprintf("Estimated Jacobian rank: %d", rank))
                fprintln(io, @sprintf("Singular value: %.100f", sigma[i]))
                fprintln(io, @sprintf("Spectral gap: %.6e",
                    sigma[i] / sigma[i+1]))
            end
            fprintln(io, @sprintf("Constrained %d of %d variables.",
                rank, param.num_variables))
            fprintln(io)

            if rank >= param.num_variables
                return true
            end

        else
            fprintln(io, "Failed to add rooted tree: ", trees[index].data)
            fprintln(io, @sprintf("Pre-refinement residual: %.6e",
                max_residual))
            fprintln(io)
            deleteat!(trees, index)
        end
    end

    return false
end


function seed_work(
    seed::UInt64,
    param::AbstractRKParameterization{T},
    param_big::AbstractRKParameterization{BigFloat},
    height_limit::Int,
    epsilon::T,
    radius::T,
) where {T}

    # TODO: Files need an indicator of being done.
    # TODO: File names should indicate satisfied order conditions.
    # TODO: A list of existing files should be built on startup.
    tempname = @sprintf("RKTK-IS-%02d-%02d-%016X.temp",
        param.num_parallel_stages + 1,
        param.parallel_width, seed)
    filename = @sprintf("RKTK-IS-%02d-%02d-%016X.txt",
        param.num_parallel_stages + 1,
        param.parallel_width, seed)
    if isfile(tempname) || isfile(filename)
        return nothing
    end

    io = open(tempname, "w")
    active_trees = LevelSequence[]
    x = random_array(seed, T, param.num_variables)
    for c in x
        fprintln(io, @sprintf("%+.100f", c))
    end
    fprintln(io)

    order = 0
    while true
        order += 1

        while true
            num_trees_old = length(active_trees)
            reached_full_rank = autosearch!(
                io, active_trees, x, param, param_big,
                order, height_limit, epsilon, radius)
            num_trees_new = length(active_trees)
            if reached_full_rank || (num_trees_old == num_trees_new)
                break
            end
        end

        x_big, max_residual, sigma, iter_count = refine_bigfloat(
            active_trees, param_big, x)
        for c in x_big
            fprintln(io, @sprintf("%+.100f", c))
        end

        fprintln(io) ###########################################################

        fprintln(io, "Processed all rooted trees of order: ", order)
        fprintln(io, "Number of active rooted trees: ",
            length(active_trees))
        @assert max_residual < 1.0e-100
        if iszero(max_residual)
            fprintln(io, "Successfully refined to exact accuracy.")
        else
            fprintln(io, @sprintf(
                "Successfully refined to %d-digit accuracy.",
                floor(Int, -log10(max_residual))))
        end
        fprintln(io, @sprintf("Refinement iteration count: %d",
            iter_count))
        copy!(x, T.(x_big))

        i = lastindex(sigma)
        while i >= 1 && sigma[i] < 1.0e-100
            i -= 1
        end
        rank = i
        if i == lastindex(sigma)
            fprintln(io, @sprintf("Estimated Jacobian rank: %d", rank))
            fprintln(io, @sprintf("Singular value: %.100f", sigma[i]))
        else
            fprintln(io, @sprintf("Estimated Jacobian rank: %d", rank))
            fprintln(io, @sprintf("Singular value: %.100f", sigma[i]))
            fprintln(io, @sprintf("Spectral gap: %.6e",
                sigma[i] / sigma[i+1]))
        end
        fprintln(io, @sprintf("Constrained %d of %d variables.",
            rank, param.num_variables))

        fprintln(io) ###########################################################

        if reached_full_rank
            fprint_error_coefficents(io, param_big, x_big)
            break
        end
    end

    close(io)
    mv(tempname, filename)
    println("Finished computing ", filename, '.')
    flush(stdout)
end


function thread_work(
    param::AbstractRKParameterization{T},
    param_big::AbstractRKParameterization{BigFloat},
    height_limit::Int,
    epsilon::T,
    radius::T,
) where {T}
    while true
        seed_work(rand(UInt64), param, param_big, height_limit, epsilon, radius)
    end
    return nothing
end


function main(
    param::AbstractRKParameterization{T},
    param_big::AbstractRKParameterization{BigFloat},
    height_limit::Int,
    epsilon::T,
    radius::T,
) where {T}
    dirname = @sprintf("RKTK-IS-%02d-%02d",
        param.num_parallel_stages + 1,
        param.parallel_width)
    if WRITE_FILE
        ensuredir(dirname)
    end
    @threads for _ = 1:nthreads()
        thread_work(param, param_big, height_limit, epsilon, radius)
    end
    return nothing
end


setprecision(BigFloat, 512)
main(
    RKParameterizationParallelExplicit{Float64x2}(
        parse(Int, ARGS[1]) - 1, parse(Int, ARGS[2])),
    RKParameterizationParallelExplicit{BigFloat}(
        parse(Int, ARGS[1]) - 1, parse(Int, ARGS[2])),
    parse(Int, ARGS[1]), Float64x2(1.0e-20), Float64x2(100.0)
)
