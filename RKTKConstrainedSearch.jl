using Base.Threads
using Printf
using RungeKuttaToolKit
using RungeKuttaToolKit.RKParameterization
using UUIDs


const USAGE_STRING = """
Usage: julia [jl_options] $PROGRAM_FILE <num_stages> [parallel_width]
[jl_options] refers to Julia options, such as -O3 or --threads=N.

<num_stages> and [parallel_width] are positive integers between 1 and 99.
If not specified, [parallel_width] defaults to 1.
"""


const EXIT_INVALID_ARGS = 1


@inline max_residual(opt::ConstrainedRKOCOptimizer) =
    maximum(abs, opt.joint_residual)


@inline max_coeff(opt::ConstrainedRKOCOptimizer) =
    max(maximum(abs, opt.A), maximum(abs, opt.b))


@inline scores(opt::ConstrainedRKOCOptimizer) = (
    maximum(abs, opt.constraint_residual),
    maximum(abs, opt.objective_residual),
    max_coeff(opt))


const VERBOSE = (stdout isa Base.TTY)
const TOTAL_ITERATION_COUNT = Atomic{Int}(0)


function constrained_search(
    active_trees::AbstractVector{LevelSequence},
    candidate_trees::AbstractVector{LevelSequence},
    param::AbstractRKParameterization{T},
    x::AbstractVector{T},
    threshold::T,
    radius::T,
) where {T}
    opt = ConstrainedRKOCOptimizer(LevelSequence[], active_trees, param, x)
    @static if VERBOSE
        @printf("| CLEANUP | %.17e | %.17e | %.17e\n", scores(opt)...)
    end
    while opt()
        @static if VERBOSE
            disable_sigint() do
                @printf("\x1b[A| CLEANUP | %.17e | %.17e | %.17e\n",
                    scores(opt)...)
            end
        end
    end
    optimizers = [
        ConstrainedRKOCOptimizer(active_trees, [tree], param, opt.x)
        for tree in candidate_trees]
    first = true
    terminated = falses(length(candidate_trees))
    converged = falses(length(candidate_trees))
    while true
        @threads :dynamic for i = 1:length(optimizers)
            if !terminated[i]
                result = optimizers[i]()
                atomic_add!(TOTAL_ITERATION_COUNT, 1)
                if any(isnan, optimizers[i].A) ||
                   any(isnan, optimizers[i].b) ||
                   any(isnan, optimizers[i].joint_residual)
                    terminated[i] = true
                elseif !(max_coeff(optimizers[i]) < radius)
                    terminated[i] = true
                end
                if !result
                    terminated[i] = true
                end
                cons_score, obj_score = optimizers[i].frontier[end]
                if (max_residual(optimizers[i]) < threshold) &&
                   (max_coeff(optimizers[i]) < radius)
                    terminated[i] = true
                    converged[i] = true
                end
            end
        end
        @static if VERBOSE
            disable_sigint() do
                if first
                    first = false
                else
                    @printf("\x1b[%dA", length(optimizers))
                end
                for i = 1:length(optimizers)
                    @printf("|%8d | %.17e | %.17e | %.17e\n",
                        i, scores(optimizers[i])...)
                end
            end
        end
        if any(converged) || all(terminated)
            break
        end
    end
    return [(i, optimizers[i], candidate_trees[i])
            for i = 1:length(candidate_trees) if converged[i]]
end


function write_rk_method(io::IO, A::Matrix{T}, b::Vector{T}) where {T}
    s = length(b)
    @assert size(A) == (s, s)
    strings = reshape([@sprintf("%+.17e", x) for x in vcat(A, b')'], s, s + 1)
    for i = 1:s+1
        for j = 1:s
            write(io, strings[j, i], j == s ? '\n' : '\t')
        end
        if i == s
            write(io, '\n')
        end
    end
end


function main(
    num_stages::Int,
    parallel_width::Int,
)
    param = RKParameterizationParallelExplicit{Float64}(
        num_stages - 1, parallel_width)
    x = rand(Float64, param.num_variables)
    uuid = uuid4()

    active_trees = all_rooted_trees(3)
    start = time_ns()

    for order = 4:99
        candidate_trees = rooted_trees(order)
        num_trees = length(candidate_trees)
        while !isempty(candidate_trees)
            finished = constrained_search(
                active_trees, candidate_trees, param, x, 1.0e-10, 100.0)
            if isempty(finished)
                disable_sigint() do
                    elapsed = (time_ns() - start) / 1.0e9
                    @printf("\nSatisfied all conditions for order %d.\n",
                        order - 1)
                    @printf("Satisfied %d of %d conditions for order %d.\n\n",
                        num_trees - length(candidate_trees), num_trees, order)
                    write_rk_method(stdout, param(x)...)
                    println()
                    filename = @sprintf("RKTK-CS-%02d-%02d-%s-%02d.txt",
                        num_stages, parallel_width,
                        uppercase(string(uuid)), order)
                    open(filename, "w") do io
                        write_rk_method(io, param(x)...)
                        @printf(io,
                            """
                            \nThis method satisfies all conditions for order %d \
                            and the following %d of %d conditions for order %d:\n
                            """,
                            order - 1, num_trees - length(candidate_trees),
                            num_trees, order)
                        for tree in active_trees
                            if length(tree) == order
                                println(io, tree.data)
                            end
                        end
                    end
                    @printf(
                        """
                        Executed %d IPOPT iterations in %.3f seconds \
                        (%.3f iterations per second).\n
                        """,
                        TOTAL_ITERATION_COUNT[], elapsed,
                        TOTAL_ITERATION_COUNT[] / elapsed)
                end
                return nothing
            end
            sort!(finished, by=((_, opt, _),) -> max_residual(opt))
            i, opt, tree = first(finished)
            println()
            println("Successfully added tree: ", tree.data)
            println()
            copy!(x, opt.x)
            push!(active_trees, tree)
            deleteat!(candidate_trees, i)
        end
        disable_sigint() do
            elapsed = (time_ns() - start) / 1.0e9
            write_rk_method(stdout, param(x)...)
            println()
            filename = @sprintf("RKTK-CS-%02d-%02d-%s-%02d.txt",
                num_stages, parallel_width, uppercase(string(uuid)), order)
            open(filename, "w") do io
                write_rk_method(io, param(x)...)
                @printf(io,
                    "\nThis method satisfies all conditions for order %d.\n",
                    order)
            end
            @printf(
                """
                Executed %d IPOPT iterations in %.3f seconds \
                (%.3f iterations per second).\n
                """,
                TOTAL_ITERATION_COUNT[], elapsed,
                TOTAL_ITERATION_COUNT[] / elapsed)
        end
    end
end


function parse_arguments()
    try
        @assert 1 <= length(ARGS) <= 2

        num_stages = parse(Int, ARGS[1])
        @assert 0 < num_stages < 100

        parallel_width = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
        @assert 0 < parallel_width < 100

        return (num_stages, parallel_width)

    catch e
        if typeof(e) in [
            ArgumentError, AssertionError, BoundsError, OverflowError
        ]
            print(stderr, USAGE_STRING)
            exit(EXIT_INVALID_ARGS)
        else
            rethrow(e)
        end
    end
end


while true
    main(parse_arguments()...)
end
