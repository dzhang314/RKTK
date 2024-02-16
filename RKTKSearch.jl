using Base.Threads
using DZOptimization
using DZOptimization.Kernels: norm2
using DZOptimization.PCG
using MultiFloats
using Printf
using RungeKuttaToolKit


const WRITE_FILE = !("--no-file" in ARGS)
const WRITE_TERM = ("--no-file" in ARGS) || (
    (stdout isa Base.TTY) && (nthreads() == 1))
filter!(arg -> (arg != "--no-file"), ARGS)


function fprintln(io::IO, args...)
    if WRITE_FILE
        println(io, args...)
        flush(io)
    else
        @assert io === devnull
    end
    if WRITE_TERM
        println(stdout, args...)
        flush(stdout)
    end
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
            sqrt(norm2(opt.current_point) / num_variables),
            sqrt(norm2(opt.delta_point) / num_variables),
            reset_occurred ? " RESET" : ""))
    end
end


function find_existing_file(prefix::AbstractString, suffix::AbstractString)
    for filename in readdir()
        if (length(filename) == 46 &&
            startswith(filename, prefix) && endswith(filename, suffix))
            return filename
        end
    end
    return nothing
end


function search(
    evaluator::RKOCEvaluator{Float64},
    order::Int, num_stages::Int, seed::UInt64
)
    tempname = @sprintf("RKTK-%02d-%02d-XXXX-XXXX-XXXX-%016X.txt",
        order, num_stages, seed)
    @assert length(tempname) == 46
    if WRITE_FILE
        existing = find_existing_file(tempname[1:11], tempname[26:46])
        if isnothing(existing)
            println("Computing $tempname...")
        else
            println("$existing already exists.")
            return nothing
        end
    end

    num_variables = ((num_stages + 1) * num_stages) >> 1
    opt = LBFGSOptimizer(evaluator, evaluator', QuadraticLineSearch(),
        random_array(seed, Float64, num_variables),
        sqrt(num_variables * eps(Float64)), num_variables)

    io = WRITE_FILE ? open(tempname, "w") : devnull

    for x in opt.current_point
        fprintln(io, @sprintf("%+.16e", x))
    end
    fprintln(io)
    fprintln(io, "| ITERATION # |  RMS RESIDUAL  |  RMS GRADIENT  " *
                 "|   RMS  COEFF   |  STEP  LENGTH  |")
    fprintln(io, "|-------------|----------------|----------------" *
                 "|----------------|----------------|")
    fprint_status(io, opt; force=true)
    while !opt.has_terminated[]
        step!(opt)
        fprint_status(io, opt;
            force=(opt.iteration_count[] % 1000 == 0))
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

    filename = @sprintf("RKTK-%02d-%02d-%04d-%04d-%04d-%016X.txt", order,
        num_stages, residual_score, gradient_score, coeff_score, seed)
    if WRITE_FILE
        mv(tempname, filename)
    end
    println("Finished computing $filename.")
end


const SEED_COUNTER = Atomic{UInt64}(0)


function thread_work(order::Int, num_stages::Int, max_seed::UInt64)
    evaluator = RKOCEvaluator{Float64}(order, num_stages)
    while true
        seed = atomic_add!(SEED_COUNTER, one(UInt64))
        if seed > max_seed
            break
        end
        search(evaluator, order, num_stages, seed)
    end
end


function main(order::Int, num_stages::Int, min_seed::UInt64, max_seed::UInt64)
    if WRITE_FILE
        dirname = @sprintf("RKTK-SEARCH-%02d-%02d", order, num_stages)
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


const USAGE_STRING = """
Usage: julia [options] $PROGRAM_FILE <order> <num_stages> [seed] [seed]
[options] refers to Julia options, such as -O3 or --math-mode=fast.
<order> and <num_stages> must be positive integers.
If no seed is specified, then a search is performed for every possible seed,
    starting from zero and counting up through every unsigned 64-bit integer.
If one seed is specified, it is treated as an upper bound (inclusive).
If two seeds are specified, they are treated as bounds (inclusive).
"""


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
