using Base.Threads
using DZOptimization
using DZOptimization.PCG
using DZOptimization.Kernels: norm2
using Printf
using RungeKuttaToolKit
using RungeKuttaToolKit.RKCost
using RungeKuttaToolKit.RKParameterization


push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using RKTKUtilities


const USAGE_STRING = """
Usage: julia [jl_options] $PROGRAM_FILE <mode> <order> <stages> [seed] [seed]
[jl_options] refers to Julia options, such as -O3 or --threads=N.

<mode> is a four-character RKTK mode string as described in the RKTK manual.
<order> and <stages> are positive integers between 1 and 99.

If no seed is specified, then a search is performed for every possible seed
    from 0 to 2^64 - 1.
If one seed is specified, it is treated as an upper bound (inclusive).
If two seeds are specified, they are treated as lower and upper bounds.
"""


const WRITE_TERM = ((stdout isa Base.TTY || stdout isa Base.PipeEndpoint) &&
                    (nthreads() == 1))
const WRITE_FILE = !("--no-file" in ARGS)
filter!(arg -> (arg != "--no-file"), ARGS)


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


function fprintln_table_header(io::IO, opt)
    fprintln(io, "| ITERATION |  COST FUNC.  | MAX RESIDUAL " *
                 "| RMS GRADIENT |  MAX COEFF.  |  STEPLENGTH  |")
    fprintln(io, "|-----------|--------------|--------------" *
                 "|--------------|--------------|--------------|")
    return nothing
end


function fprintln_table_row(io::IO, opt)
    fprintln(io, @sprintf("|%10d | %.6e | %.6e | %.6e | %.6e | %.6e |%s",
        opt.iteration_count[],
        opt.current_objective_value[],
        compute_max_residual(opt),
        compute_rms_gradient(opt),
        compute_max_coeff(opt),
        sqrt(norm2(opt.delta_point)),
        reset_occurred(opt) ? " RESET" : ""))
    return nothing
end


const TOTAL_ITERATION_COUNT = Atomic{Int}(0)


function search(
    seed::UInt64,
    prefix::AbstractString,
    prob::RKOCOptimizationProblem,
)
    opt = construct_optimizer(seed, prob)
    filename = @sprintf("%s-XXXX-XXXX-XXXX-%016X.txt", prefix, seed)

    @static if WRITE_FILE
        io = open(filename, "w")
    else
        io = devnull
    end

    ######################################################## PRINT INITIAL POINT

    for line in uniform_precision_strings(opt.current_point)
        fprintln(io, line)
    end

    fprintln(io) ################################# RUN OPTIMIZER AND PRINT TABLE

    fprintln_table_header(io, opt)
    fprintln_table_row(io, opt)
    failed = false
    start_time = time_ns()
    while true
        step!(opt)
        if opt.has_terminated[]
            break
        end
        if !(compute_max_coeff(opt) < 100.0)
            failed = true
            break
        end
        if reset_occurred(opt) || (opt.iteration_count[] % 1000 == 0)
            fprintln_table_row(io, opt)
        end
    end
    end_time = time_ns()
    fprintln_table_row(io, opt)

    fprintln(io) ############################################# PRINT FINAL POINT

    for line in uniform_precision_strings(opt.current_point)
        fprintln(io, line)
    end

    ########################################################## CREATE FINAL FILE

    finalname = @sprintf("%s-%04d-%04d-%s-%016X.txt",
        prefix, compute_residual_score(opt), compute_gradient_score(opt),
        failed ? "FAIL" : @sprintf("%04d", compute_coeff_score(opt)), seed)

    atomic_add!(TOTAL_ITERATION_COUNT, opt.iteration_count[])

    if WRITE_FILE
        close(io)
        mv(filename, finalname)
        elapsed_time = (end_time - start_time) / 1.0e9
        @printf("Finished computing %s.\nPerformed %d L-BFGS iterations in %g seconds (%g iterations per second).\n",
            finalname, opt.iteration_count[], elapsed_time, opt.iteration_count[] / elapsed_time)
    end
end


const SEED_COUNTER = Atomic{UInt64}(0)


function thread_work(
    prefix::AbstractString,
    order::Int,
    param::AbstractRKParameterization{T},
    max_seed::UInt64,
) where {T}
    prob = RKOCOptimizationProblem(
        RKOCEvaluator{T}(order, param.num_stages),
        RKCostL2{T}(), param)
    while true
        seed = atomic_add!(SEED_COUNTER, one(UInt64))
        if seed > max_seed
            break
        end
        search(seed, prefix, prob)
    end
end


function main(
    mode::AbstractString,
    order::Int,
    stages::Int,
    min_seed::UInt64,
    max_seed::UInt64,
)
    dirname = @sprintf("RKTK-%02d-%02d-%s", order, stages, mode[1:2])
    prefix = @sprintf("RKTK-%02d-%02d-%s", order, stages, mode)
    if WRITE_FILE
        ensuredir(dirname)
    end
    SEED_COUNTER[] = min_seed
    start_time = time_ns()
    @threads for _ = 1:nthreads()
        param = compute_parameterization(ARGS[1], stages)
        thread_work(prefix, order, param, max_seed)
    end
    end_time = time_ns()
    elapsed_time = (end_time - start_time) / 1.0e9
    if WRITE_FILE
        @printf("In total, performed %d L-BFGS iterations in %g seconds (%g iterations per second).\n",
            TOTAL_ITERATION_COUNT[], elapsed_time, TOTAL_ITERATION_COUNT[] / elapsed_time)
    end
end


function parse_arguments()
    try
        @assert 3 <= length(ARGS) <= 5

        mode = ARGS[1]
        @assert is_valid_mode(mode)

        order = parse(Int, ARGS[2])
        @assert 0 < order < 100

        stages = parse(Int, ARGS[3])
        @assert 0 < stages < 100

        min_seed = typemin(UInt64)
        max_seed = typemax(UInt64)
        if length(ARGS) == 4
            max_seed = parse(UInt64, ARGS[4])
        elseif length(ARGS) == 5
            min_seed = parse(UInt64, ARGS[4])
            max_seed = parse(UInt64, ARGS[5])
        end
        @assert min_seed <= max_seed

        return (mode, order, stages, min_seed, max_seed)

    catch e
        if typeof(e) in [
            ArgumentError, AssertionError, BoundsError, OverflowError
        ]
            print(USAGE_STRING)
            exit(EXIT_INVALID_ARG_FORMAT)
        else
            rethrow(e)
        end
    end
end


main(parse_arguments()...)
