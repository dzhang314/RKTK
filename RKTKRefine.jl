using DZOptimization
using DZOptimization.Kernels: norm2
using MultiFloats
using Printf
using RungeKuttaToolKit


const USAGE_STRING = """
Usage: julia [jl_options] $PROGRAM_FILE <mode> [order] [stages] <filename>
[jl_options] refers to Julia options, such as -O3 or --threads=8.

<mode> is a four-character RKTK mode string, as explained in the RKTK manual.
[order] and [stages] must be positive integers between 1 and 99.
If <filename> is an RKTK file, then [order] and [stages] can be omitted.
"""


if (length(ARGS) != 2) && (length(ARGS) != 4)
    print(stderr, USAGE_STRING)
    exit(EXIT_INVALID_ARG_COUNT)
end


include("./src/ParseMode.jl")


function fprintln(io::IO, args...)
    println(io, args...)
    flush(io)
    println(stdout, args...)
    flush(stdout)
    return nothing
end


function fprint_status(io::IO, opt::RKOCOptimizer; force::Bool=false)
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


const RKTK_TXT_FILENAME_REGEX =
    r"^RKTK-([0-9]{2})-([0-9]{2})-([AB][EDI][MA][0-9])-([0-9]{4})-([0-9]{4})-([0-9]{4}|FAIL)-([0-9A-Fa-f]{16})\.txt$"


function main_rktk_file(filename::String)

    m = match(RKTK_TXT_FILENAME_REGEX, basename(filename))
    @assert !isnothing(m)

    order = parse(Int, m[1]; base=10)
    stages = parse(Int, m[2]; base=10)
    seed = parse(UInt64, m[7]; base=16)

    n = num_parameters(stages)
    initial_point = parse_last_block(filename, stages)
    for line in uniform_precision_strings(initial_point)
        println(line)
    end

    evaluator = RKOCEvaluator(order, stages)
    opt = LBFGSOptimizer(evaluator, evaluator', QuadraticLineSearch(),
        initial_point, sqrt(n * eps(T)), n)

    io = open(filename, "a")
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
        if (any(!(abs(c) <= 1024.0) for c in opt.objective_function.A) ||
            any(!(abs(c) <= 1024.0) for c in opt.objective_function.b))
            failed = true
            break
        end
        fprint_status(io, opt; force=(opt.iteration_count[] % 100 == 0))
    end
    fprint_status(io, opt; force=true)

    fprintln(io)

    for line in uniform_precision_strings(opt.current_point)
        fprintln(io, line)
    end

    close(io)

    residual_score, gradient_score, coeff_score = compute_scores(opt)
    finalname = @sprintf("RKTK-%02d-%02d-%s%s-%04d-%04d-%s-%016X.txt",
        order, stages, PARAMETERIZATION, PRECISION, residual_score,
        gradient_score, failed ? "FAIL" : @sprintf("%04d", coeff_score), seed)

    mv(filename, finalname)

    println("Wrote results to: $finalname")
end


function main(order::Int, stages::Int, filename::String)

    n = num_parameters(stages)
    initial_point = parse_last_block(filename, stages)
    for line in uniform_precision_strings(initial_point)
        println(line)
    end

    evaluator = RKOCEvaluator(order, stages)
    opt = LBFGSOptimizer(evaluator, evaluator', QuadraticLineSearch(),
        initial_point, sqrt(n * eps(T)), n)

    io = open(filename, "a")
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
        if (any(!(abs(c) <= 1024.0) for c in opt.objective_function.A) ||
            any(!(abs(c) <= 1024.0) for c in opt.objective_function.b))
            failed = true
            break
        end
        fprint_status(io, opt; force=(opt.iteration_count[] % 100 == 0))
    end
    fprint_status(io, opt; force=true)

    fprintln(io)

    for x in opt.current_point
        fprintln(io, x)
    end

    close(io)
    println("Wrote results to: $filename")
end


function parse_arguments()
    try
        @assert 3 <= length(ARGS) <= 5
        order = parse(Int, ARGS[2])
        @assert 0 < order < 100
        stages = parse(Int, ARGS[3])
        @assert 0 < stages < 100
        filename = ARGS[4]
        return (order, stages, filename)
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


warn_single_threaded()
if !isnothing(match(RKTK_TXT_FILENAME_REGEX, basename(ARGS[2])))
    main_rktk_file(ARGS[2])
else
    main(parse_arguments()...)
end
