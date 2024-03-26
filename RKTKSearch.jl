using Base.Threads
using DZOptimization
using DZOptimization.Kernels: norm2
using DZOptimization.PCG
using MultiFloats
using Printf
using RungeKuttaToolKit


const EXIT_INVALID_ARG_COUNT = 1
const EXIT_INVALID_ARG_FORMAT = 2
const EXIT_INVALID_MODE_LENGTH = 3
const EXIT_INVALID_PARAMETERIZATION = 4
const EXIT_INVALID_PRECISION = 5


const USAGE_STRING = """
Usage: julia [jl_options] $PROGRAM_FILE <mode> <order> <stages> [seed] [seed]
[jl_options] refers to Julia options, such as -O3 or --threads=8.

<mode> is a four-character RKTK mode string, as explained in the RKTK manual.
<order> and <stages> must be positive integers between 1 and 99.

If no seed is specified, then a search is performed for every possible seed,
    starting from zero and counting up through every unsigned 64-bit integer.
If one seed is specified, it is treated as an upper bound (inclusive).
If two seeds are specified, they are treated as interval bounds (inclusive).
"""


const WRITE_FILE = !("--no-file" in ARGS)
const WRITE_TERM = ("--no-file" in ARGS) || (
    (stdout isa Base.TTY) && (nthreads() == 1))
filter!(arg -> (arg != "--no-file"), ARGS)


if (length(ARGS) < 3) || (length(ARGS) > 5)
    print(stderr, USAGE_STRING)
    exit(EXIT_INVALID_ARG_COUNT)
end


include("./src/ParseMode.jl")


function fprintln(io::IO, args...)
    @static if WRITE_FILE
        println(io, args...)
        flush(io)
    end
    @static if WRITE_TERM
        println(stdout, args...)
        flush(stdout)
    end
    return nothing
end


function fprint_status(io::IO, opt::LBFGSOptimizer; force::Bool=false)
    history_count = length(opt._rho)
    reset_occurred = ((opt.iteration_count[] >= history_count) &&
                      (opt._history_count[] != history_count))
    if reset_occurred || force
        num_residuals = length(opt.objective_function.residuals)
        n = length(opt.current_point)
        fprintln(io, @sprintf("|%12d | %.8e | %.8e | %.8e | %.8e |%s",
            opt.iteration_count[],
            sqrt(opt.current_objective_value[] / num_residuals),
            sqrt(norm2(opt.current_gradient) / n),
            sqrt(maximum(abs, opt.current_point)),
            sqrt(norm2(opt.delta_point) / n),
            reset_occurred ? " RESET" : ""))
    end
    return nothing
end


function find_existing_file(prefix::AbstractString, suffix::AbstractString)
    for filename in readdir()
        if (length(filename) == 51 &&
            startswith(filename, prefix) && endswith(filename, suffix))
            return filename
        end
    end
    return nothing
end


const TOTAL_ITERATION_COUNT = Atomic{Int}(0)


function search(
    evaluator::RKOCEvaluator, order::Int, stages::Int, seed::UInt64
)
    filename = @sprintf("RKTK-%02d-%02d-%s%s-XXXX-XXXX-XXXX-%016X.txt",
        order, stages, PARAMETERIZATION, PRECISION, seed)
    @assert length(filename) == 51

    if WRITE_FILE
        existing = find_existing_file(filename[1:16], filename[31:51])
        if isnothing(existing)
            @printf("Computing %s...\n", filename)
        else
            @printf("%s already exists.\n", existing)
            return nothing
        end
    end

    n = num_parameters(stages)
    opt = LBFGSOptimizer(evaluator, evaluator', QuadraticLineSearch(),
        random_array(seed, T, n), sqrt(n * eps(T)), n)

    @static if WRITE_FILE
        io = open(filename, "w")
    else
        io = devnull
    end

    if T == Float64
        for x in opt.current_point
            fprintln(io, @sprintf("%+.16e", x))
        end
    else
        for x in opt.current_point
            fprintln(io, x)
        end
    end

    fprintln(io)

    fprintln(io, "| ITERATION # |  RMS RESIDUAL  |  RMS GRADIENT  " *
                 "| MAX ABS COEFF. |  STEP  LENGTH  |")
    fprintln(io, "|-------------|----------------|----------------" *
                 "|----------------|----------------|")
    fprint_status(io, opt; force=true)
    failed = false
    start_time = time_ns()
    while true
        step!(opt)
        if opt.has_terminated[]
            break
        end
        if (any(!(abs(c) <= 1024.0) for c in opt.objective_function.A) ||
            any(!(abs(c) <= 1024.0) for c in opt.objective_function.b))
            failed = true
            break
        end
        fprint_status(io, opt; force=(opt.iteration_count[] % 1000 == 0))
    end
    end_time = time_ns()
    fprint_status(io, opt; force=true)

    fprintln(io)

    if T == Float64
        for x in opt.current_point
            fprintln(io, @sprintf("%+.16e", x))
        end
    else
        for x in opt.current_point
            fprintln(io, x)
        end
    end

    if WRITE_FILE
        close(io)
    end

    num_residuals = length(evaluator.residuals)
    rms_residual = sqrt(Float64(opt.current_objective_value[]) / num_residuals)
    rms_gradient = sqrt(Float64(norm2(opt.current_gradient)) / n)
    rms_coeff = sqrt(Float64(norm2(opt.current_point)) / n)
    residual_score = round(Int, clamp(-500 * log10(rms_residual), 0.0, 9999.0))
    gradient_score = round(Int, clamp(-500 * log10(rms_gradient), 0.0, 9999.0))
    coeff_score = round(Int, clamp(10000 - 2500 * log10(rms_coeff), 0.0, 9999.0))

    finalname = @sprintf("RKTK-%02d-%02d-%s%s-%04d-%04d-%s-%016X.txt",
        order, stages, PARAMETERIZATION, PRECISION, residual_score,
        gradient_score, failed ? "FAIL" : @sprintf("%04d", coeff_score), seed)
    if WRITE_FILE
        mv(filename, finalname)
    end

    elapsed_time = (end_time - start_time) / 1.0e9
    atomic_add!(TOTAL_ITERATION_COUNT, opt.iteration_count[])
    @printf("Finished computing %s.\nPerformed %d L-BFGS iterations in %g seconds (%g iterations per second).\n",
        finalname, opt.iteration_count[], elapsed_time, opt.iteration_count[] / elapsed_time)
end


const SEED_COUNTER = Atomic{UInt64}(0)


function thread_work(order::Int, stages::Int, max_seed::UInt64)
    evaluator = RKOCEvaluator(order, stages)
    while true
        seed = atomic_add!(SEED_COUNTER, one(UInt64))
        if seed > max_seed
            break
        end
        search(evaluator, order, stages, seed)
    end
end


function main(order::Int, stages::Int, min_seed::UInt64, max_seed::UInt64)
    if WRITE_FILE
        dirname = @sprintf("RKTK-%02d-%02d-%s", order, stages, PARAMETERIZATION)
        if basename(pwd()) != dirname
            if !isdir(dirname)
                mkdir(dirname)
            end
            cd(dirname)
        end
    end
    SEED_COUNTER[] = min_seed
    start_time = time_ns()
    @threads for _ = 1:nthreads()
        thread_work(order, stages, max_seed)
    end
    end_time = time_ns()
    elapsed_time = (end_time - start_time) / 1.0e9
    @printf("In total, performed %d L-BFGS iterations in %g seconds (%g iterations per second).\n",
        TOTAL_ITERATION_COUNT[], elapsed_time, TOTAL_ITERATION_COUNT[] / elapsed_time)
end


function parse_arguments()
    try
        @assert 3 <= length(ARGS) <= 5
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
        return (order, stages, min_seed, max_seed)
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
