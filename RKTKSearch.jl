using Base.Threads
using DZOptimization
using DZOptimization.Kernels: norm2
using DZOptimization.PCG
using MultiFloats
using Printf
using RungeKuttaToolKit


const USAGE_STRING = """
Usage: julia [jl_options] $PROGRAM_FILE <order> <num_stages> [seed] [seed]
[jl_options] refers to Julia options, such as -O3 or --math-mode=fast.

<order> and <num_stages> must be positive integers between 1 and 99.

If no seed is specified, then a search is performed for every possible seed,
    starting from zero and counting up through every unsigned 64-bit integer.
If one seed is specified, it is treated as an upper bound (inclusive).
If two seeds are specified, they are treated as bounds (inclusive).
"""


const WRITE_FILE = !("--no-file" in ARGS)
const WRITE_TERM = ("--no-file" in ARGS) || (
    (stdout isa Base.TTY) && (nthreads() == 1))
const REQUESTED_EXPLICIT = ("--explicit" in ARGS)
const REQUESTED_IMPLICIT = ("--implicit" in ARGS)
const REQUESTED_EXACT_B = ("--exact-b" in ARGS)
const REQUESTED_APPROXIMATE_B = ("--approximate-b" in ARGS)
filter!(arg -> (
        (arg != "--no-file") && (arg != "--explicit") &&
        (arg != "--implicit") && (arg != "--exact-b") &&
        (arg != "--approximate-b")), ARGS)


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
        num_variables = length(opt.current_point)
        fprintln(io, @sprintf("|%12d | %.8e | %.8e | %.8e | %.8e |%s",
            opt.iteration_count[],
            sqrt(opt.current_objective_value[] / num_residuals),
            sqrt(norm2(opt.current_gradient) / num_variables),
            sqrt(maximum(abs, opt.current_point)),
            sqrt(norm2(opt.delta_point) / num_variables),
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


function search(
    evaluator, order::Int, num_stages::Int, mode::String, seed::UInt64
)
    @assert length(mode) == 4
    filename = @sprintf("RKTK-%02d-%02d-%s-XXXX-XXXX-XXXX-%016X.txt",
        order, num_stages, mode, seed)
    @assert length(filename) == 51

    if WRITE_FILE
        existing = find_existing_file(filename[1:16], filename[31:51])
        if isnothing(existing)
            println("Computing $filename...")
        else
            println("$existing already exists.")
            return nothing
        end
    end

    num_variables =
        (mode == "AEM1") ? (num_stages * (num_stages - 1)) >> 1 :
        (mode == "BEM1") ? (num_stages * (num_stages + 1)) >> 1 :
        (mode == "AIM1") ? num_stages * num_stages :
        (mode == "BIM1") ? num_stages * (num_stages + 1) : 0
    opt = LBFGSOptimizer(evaluator, evaluator', QuadraticLineSearch(),
        random_array(seed, Float64, num_variables),
        sqrt(num_variables * eps(Float64)), num_variables)

    io = WRITE_FILE ? open(filename, "w") : devnull

    for x in opt.current_point
        fprintln(io, @sprintf("%+.16e", x))
    end

    fprintln(io)

    fprintln(io, "| ITERATION # |  RMS RESIDUAL  |  RMS GRADIENT  " *
                 "| MAX ABS COEFF. |  STEP  LENGTH  |")
    fprintln(io, "|-------------|----------------|----------------" *
                 "|----------------|----------------|")
    fprint_status(io, opt; force=true)
    failed = false
    while true
        step!(opt)
        if opt.has_terminated[]
            break
        end
        if any(abs(coeff) > 1024.0 for coeff in opt.current_point)
            failed = true
            break
        end
        fprint_status(io, opt; force=(opt.iteration_count[] % 1000 == 0))
    end
    fprint_status(io, opt; force=true)

    fprintln(io)

    for x in opt.current_point
        fprintln(io, @sprintf("%+.16e", x))
    end

    if WRITE_FILE
        close(io)
    end

    num_residuals = length(evaluator.residuals)
    rms_residual = sqrt(opt.current_objective_value[] / num_residuals)
    rms_gradient = sqrt(norm2(opt.current_gradient) / num_variables)
    rms_coeff = sqrt(norm2(opt.current_point) / num_variables)
    residual_score = round(Int,
        clamp(-500 * log10(rms_residual), 0.0, 9999.0))
    gradient_score = round(Int,
        clamp(-500 * log10(rms_gradient), 0.0, 9999.0))
    coeff_score = round(Int,
        clamp(10000 - 2500 * log10(rms_coeff), 0.0, 9999.0))

    finalname = @sprintf("RKTK-%02d-%02d-%s-%04d-%04d-%s-%016X.txt",
        order, num_stages, mode, residual_score, gradient_score,
        failed ? "FAIL" : @sprintf("%04d", coeff_score), seed)
    if WRITE_FILE
        mv(filename, finalname)
    end
    println("Finished computing $finalname.")
end


const SEED_COUNTER = Atomic{UInt64}(0)


function get_mode()
    @static if REQUESTED_IMPLICIT && !REQUESTED_EXPLICIT
        @static if REQUESTED_APPROXIMATE_B && !REQUESTED_EXACT_B
            return "BI"
        else
            return "AI"
        end
    else
        @static if REQUESTED_APPROXIMATE_B && !REQUESTED_EXACT_B
            return "BE"
        else
            return "AE"
        end
    end
end


function thread_work(order::Int, num_stages::Int, max_seed::UInt64)
    @static if REQUESTED_IMPLICIT && !REQUESTED_EXPLICIT
        @static if REQUESTED_APPROXIMATE_B && !REQUESTED_EXACT_B
            evaluator = RKOCEvaluatorBI{Float64}(order, num_stages)
            while true
                seed = atomic_add!(SEED_COUNTER, one(UInt64))
                if seed > max_seed
                    break
                end
                search(evaluator, order, num_stages, "BIM1", seed)
            end
        else
            evaluator = RKOCEvaluatorAI{Float64}(order, num_stages)
            while true
                seed = atomic_add!(SEED_COUNTER, one(UInt64))
                if seed > max_seed
                    break
                end
                search(evaluator, order, num_stages, "AIM1", seed)
            end
        end
    else
        @static if REQUESTED_APPROXIMATE_B && !REQUESTED_EXACT_B
            evaluator = RKOCEvaluatorBE{Float64}(order, num_stages)
            while true
                seed = atomic_add!(SEED_COUNTER, one(UInt64))
                if seed > max_seed
                    break
                end
                search(evaluator, order, num_stages, "BEM1", seed)
            end
        else
            evaluator = RKOCEvaluatorAE{Float64}(order, num_stages)
            while true
                seed = atomic_add!(SEED_COUNTER, one(UInt64))
                if seed > max_seed
                    break
                end
                search(evaluator, order, num_stages, "AEM1", seed)
            end
        end
    end
end


function main(order::Int, num_stages::Int, min_seed::UInt64, max_seed::UInt64)
    if WRITE_FILE
        dirname = @sprintf("RKTK-%02d-%02d-%s", order, num_stages, get_mode())
        if basename(pwd()) != dirname
            if !isdir(dirname)
                mkdir(dirname)
            end
            cd(dirname)
        end
    end
    SEED_COUNTER[] = min_seed
    @threads for _ = 1:nthreads()
        thread_work(order, num_stages, max_seed)
    end
end


function parse_arguments()
    try
        @assert 2 <= length(ARGS) <= 4
        order = parse(Int, ARGS[1])
        @assert 0 < order < 100
        num_stages = parse(Int, ARGS[2])
        @assert 0 < num_stages < 100
        min_seed = typemin(UInt64)
        max_seed = typemax(UInt64)
        if length(ARGS) == 3
            max_seed = parse(UInt64, ARGS[3])
        elseif length(ARGS) == 4
            min_seed = parse(UInt64, ARGS[3])
            max_seed = parse(UInt64, ARGS[4])
        end
        @assert min_seed <= max_seed
        return (order, num_stages, min_seed, max_seed)
    catch e
        if typeof(e) in [
            ArgumentError, AssertionError, BoundsError, OverflowError
        ]
            print(USAGE_STRING)
            exit(1)
        else
            rethrow(e)
        end
    end
end


main(parse_arguments()...)
