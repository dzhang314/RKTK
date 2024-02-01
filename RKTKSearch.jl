using DZOptimization
using DZOptimization: norm2
using DZOptimization.PCG
using MultiFloats
using Printf
using RungeKuttaToolKit


const IS_CONSOLE = (stdout isa Base.TTY)


function fprintln(io::IO, args...)
    println(io, args...)
    flush(io)
    if IS_CONSOLE
        println(stdout, args...)
        flush(stdout)
    end
end


function fprint_status(io::IO, opt; force::Bool=false)
    history_count = length(opt._rho)
    reset_occurred = ((opt.iteration_count[] >= history_count) &&
                      (opt._history_count[] != history_count))
    if reset_occurred || force
        num_constraints = length(opt.objective_function.output_indices)
        num_variables = length(opt.current_point)
        fprintln(io, @sprintf("|%12d | %.8e | %.8e | %.8e | %.8e |%s",
            opt.iteration_count[],
            sqrt(opt.current_objective_value[] / num_constraints),
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


function main(order::Int, num_stages::Int, min_seed::UInt64, max_seed::UInt64)

    evaluator = RKOCEvaluator{Float64}(order, num_stages)
    num_variables = ((num_stages + 1) * num_stages) >> 1
    history_count = num_variables # TODO

    for seed = min_seed:max_seed

        tempname = @sprintf("RKTK-%02d-%02d-XXXX-XXXX-XXXX-%016X.txt",
            order, num_stages, seed)
        @assert length(tempname) == 46
        existing = find_existing_file(tempname[1:11], tempname[26:46])
        if isnothing(existing)
            println("Computing $tempname...")
        else
            println("$existing already exists.")
            continue
        end

        opt = LBFGSOptimizer(evaluator, evaluator', QuadraticLineSearch(),
            random_array(seed, Float64, num_variables),
            sqrt(num_variables * eps(Float64)), history_count)

        open(tempname, "w") do io
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
        end

        num_constraints = length(evaluator.output_indices)
        rms_residual = sqrt(opt.current_objective_value[] / num_constraints)
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
        mv(tempname, filename)
        println("Finished computing $filename.")
    end
end


const USAGE_STRING = """\
Usage: julia [options] $PROGRAM_FILE <order> <num_stages> <seed> [seed]
[options] refers to Julia options, such as -O3 or --math-mode=fast.
<order> and <num_stages> must be positive integers.
If two seeds are specified, they are treated as the bounds of a range.
"""


function parse_arguments()
    try
        @assert 3 <= length(ARGS) <= 4
        order = parse(Int, ARGS[1])
        @assert order > 0
        num_stages = parse(Int, ARGS[2])
        @assert num_stages > 0
        min_seed = parse(UInt64, ARGS[3])
        max_seed = parse(UInt64, ARGS[length(ARGS) == 4 ? 4 : 3])
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
