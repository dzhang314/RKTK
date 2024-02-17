include("RKTKUtilities.jl")
using DZOptimization
using DZOptimization.Kernels: norm2
using MultiFloats
using Printf
using RungeKuttaToolKit
using Serialization


function print_status(opt::LBFGSOptimizer; force::Bool=false)
    history_count = length(opt._rho)
    reset_occurred = ((opt.iteration_count[] >= history_count) &&
                      (opt._history_count[] != history_count))
    if reset_occurred || force
        num_residuals = length(opt.objective_function.residuals)
        num_variables = length(opt.current_point)
        @printf("|%12d | %.8e | %.8e | %.8e | %.8e |%s\n",
            opt.iteration_count[],
            sqrt(opt.current_objective_value[] / num_residuals),
            sqrt(norm2(opt.current_gradient) / num_variables),
            sqrt(norm2(opt.current_point) / num_variables),
            sqrt(norm2(opt.delta_point) / num_variables),
            reset_occurred ? " RESET" : "")
    end
end


function read_rktk_search_file(filepath::AbstractString)
    m = match(RKTK_COMPLETE_FILENAME_REGEX, basename(filepath))
    @assert !isnothing(m)
    order = parse(Int, m[1]; base=10)
    num_stages = parse(Int, m[2]; base=10)
    id = parse(UInt64, m[6]; base=16)

    @assert isfile(filepath)
    parts = split(read(filepath, String), "\n\n")
    @assert length(parts) == 3
    initial_part, table, final_part = parts

    initial_lines = split(initial_part, '\n')
    final_lines = split(final_part, '\n')
    @assert length(initial_lines) + 1 == length(final_lines)
    @assert isempty(final_lines[end])
    initial_point = parse.(Float64, initial_lines)
    final_point = parse.(Float64, final_lines[1:end-1])

    table_entries = split(split(table, '\n')[end], '|')
    @assert length(table_entries) == 7
    @assert isempty(table_entries[1])
    iteration_count = parse(Int, table_entries[2])

    return (order, num_stages, id, initial_point, final_point, iteration_count)
end


const NORM_LIMIT = Float64x2(10.0)


function main()

    @assert length(ARGS) == 1
    (order, num_stages, id, _, final_point, _) = read_rktk_search_file(ARGS[1])

    evaluator = RKOCEvaluator{Float64x2}(order, num_stages)
    opt = LBFGSOptimizer(evaluator, evaluator', QuadraticLineSearch(),
        Float64x2.(final_point), sqrt(length(final_point) * eps(Float64x2)),
        length(final_point))
    norm_bound = NORM_LIMIT * NORM_LIMIT * length(opt.current_point)

    print_status(opt; force=true)
    checkpoint = @sprintf("RKTK-REFINE-%02d-%02d-%016X-%012d.jls",
        order, num_stages, id, opt.iteration_count[])
    serialize(checkpoint, opt)
    while !opt.has_terminated[]
        if norm2(opt.current_point) > norm_bound
            break
        end
        step!(opt)
        if opt.iteration_count[] % 1000 == 0
            print_status(opt; force=true)
            rm(checkpoint)
            checkpoint = @sprintf("RKTK-REFINE-%02d-%02d-%016X-%012d.jls",
                order, num_stages, id, opt.iteration_count[])
            serialize(checkpoint, opt)
        else
            print_status(opt; force=false)
        end
    end
    print_status(opt; force=true)
    rm(checkpoint)
    checkpoint = @sprintf("RKTK-REFINE-%02d-%02d-%016X-%012d.jls",
        order, num_stages, id, opt.iteration_count[])
    serialize(checkpoint, opt)

end


main()
