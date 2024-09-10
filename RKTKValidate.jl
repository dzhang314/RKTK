using Base.Threads
using DZOptimization.PCG
using Printf
using RungeKuttaToolKit
using RungeKuttaToolKit.RKCost


push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using RKTKUtilities


const USAGE_STRING = """
Usage: julia [jl_options] $PROGRAM_FILE [options]
[jl_options] refers to Julia options, such as -O3 or --threads=N.

Validate all RKTK files in the current directory.

Options:
    --recurse           Recursively validate RKTK files in subdirectories.
    --verbose           Print names of successfully validated RKTK files.
    --ignore            Ignore non-RKTK files.
    --delete-incomplete Delete incomplete RKTK files.
    --delete-invalid    Delete invalid RKTK files.
"""


const RECURSE = get_flag!(["recurse"])
const VERBOSE = get_flag!(["verbose"])
const IGNORE = get_flag!(["ignore"])
const DELETE_INCOMPLETE = get_flag!(["delete-incomplete"])
const DELETE_INVALID = get_flag!(["delete-invalid"])


function assert_rktk_file_valid(m::RegexMatch)
    @assert m.regex == RKTK_FILENAME_REGEX

    order = parse(Int, m[1]; base=10)
    stages = parse(Int, m[2]; base=10)
    mode = m[3]
    residual_score = parse(Int, m[4]; base=10)
    gradient_score = parse(Int, m[5]; base=10)
    coeff_score = m[6] == "FAIL" ? -1 : parse(Int, m[6])
    seed = parse(UInt64, m[7]; base=16)

    T = get_type(mode)
    param = get_parameterization(mode, stages)
    prob = RKOCOptimizationProblem(
        RKOCEvaluator{T}(order, param.num_stages),
        RKCostL2{T}(), param)

    blocks = split(read(m.match, String), "\n\n")
    @assert length(blocks) == 3
    @assert !startswith(blocks[1], '\n')
    @assert !endswith(blocks[1], '\n')
    @assert !startswith(blocks[2], '\n')
    @assert !endswith(blocks[2], '\n')
    @assert !startswith(blocks[3], '\n')
    @assert endswith(blocks[3], '\n')

    initial = parse_floats(T, blocks[1])
    @assert initial == random_array(seed, T, param.num_variables)

    table = split(strip(blocks[2], '\n'), '\n')
    @assert length(table) >= 4
    @assert table[1] == TABLE_HEADER
    @assert table[2] == TABLE_SEPARATOR

    table = ParsedRKTKTableRow.(table[3:end])
    @assert issorted(row.iteration for row in table)
    @assert issorted(row.cost_func for row in table; rev=true)
    @assert all(!signbit(row.max_residual) for row in table)
    @assert all(!signbit(row.rms_gradient) for row in table)
    @assert all(!signbit(row.max_coeff) for row in table)
    @assert all(!signbit(row.steplength) for row in table)

    final = parse_floats(T, blocks[3])
    @assert length(final) == param.num_variables

    opt_initial = construct_optimizer(prob, initial)
    initial_row = ParsedRKTKTableRow(compute_table_row(opt_initial))
    opt_final = construct_optimizer(prob, final)
    final_row = ParsedRKTKTableRow(compute_table_row(opt_final))
    @assert full_match(initial_row, table[1])
    @assert partial_match(final_row, table[end])

    @assert compute_residual_score(opt_final) == residual_score
    @assert compute_gradient_score(opt_final) == gradient_score
    if coeff_score == -1
        @assert compute_coeff_score(opt_final) <= 5000
    else
        @assert compute_coeff_score(opt_final) == coeff_score
    end

    return nothing
end


const VALID_FILE_COUNTER = Atomic{UInt64}(0)


function process_file(filename::AbstractString)
    @assert isfile(filename)

    m = match(RKTK_INCOMPLETE_FILENAME_REGEX, filename)
    if !isnothing(m)
        @static if DELETE_INCOMPLETE
            rm(filename)
            println("Deleted incomplete RKTK file: $filename")
        else
            println(stderr, "Found incomplete RKTK file: $filename")
        end
        return nothing
    end

    m = match(RKTK_FILENAME_REGEX, filename)
    if isnothing(m)
        @static if !IGNORE
            println(stderr, "Found non-RKTK file: $filename")
        end
        return nothing
    end

    try
        assert_rktk_file_valid(m)
        atomic_add!(VALID_FILE_COUNTER, one(UInt64))
        @static if VERBOSE
            println("Successfully validated RKTK file: $filename")
        end
    catch e
        if typeof(e) in [AssertionError, ArgumentError]
            @static if DELETE_INVALID
                rm(filename)
                println("Deleted invalid RKTK file: $filename")
            else
                println(stderr, "Failed to validate RKTK file: $filename")
                println(stderr, e)
            end
        else
            rethrow(e)
        end
    end
    return nothing
end


function process_directory()
    if all(isfile, readdir())
        # Process leaf directories in parallel.
        @threads :dynamic for filename in readdir()
            process_file(filename)
        end
    else
        # Process subdirectories in serial.
        for filename in readdir()
            if isdir(filename)
                @static if RECURSE
                    current_directory = pwd()
                    cd(filename)
                    process_directory()
                    cd(current_directory)
                end
            else
                process_file(filename)
            end
        end
    end
    return nothing
end


function main()
    if length(ARGS) != 0
        println(stderr, USAGE_STRING)
        exit(EXIT_INVALID_ARGS)
    end
    start_time = time_ns()
    process_directory()
    end_time = time_ns()
    elapsed_time = (end_time - start_time) / 1.0e9
    @printf("Successfully validated %d RKTK files in %g seconds (%g files per second).\n",
        VALID_FILE_COUNTER[], elapsed_time, VALID_FILE_COUNTER[] / elapsed_time)
end


main()
