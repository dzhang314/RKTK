using Base.Threads
using DZOptimization.PCG
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


function validate_rktk_file(filename::AbstractString)
    if !isnothing(match(RKTK_INCOMPLETE_FILENAME_REGEX, filename))
        @static if DELETE_INCOMPLETE
            rm(filename)
            println("Deleted incomplete RKTK file: $filename")
        else
            println(stderr, "Found incomplete RKTK file: $filename")
        end
    else
        m = match(RKTK_FILENAME_REGEX, filename)
        if isnothing(m)
            @static if !IGNORE
                println(stderr, "Found non-RKTK file: $filename")
            end
        else
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

            blocks = split(read(filename, String), "\n\n")
            @assert length(blocks) == 3

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

            @static if VERBOSE
                println("Successfully validated RKTK file: $filename")
            end
        end
    end
end


function validate_rktk_files()
    @threads :dynamic for filename in readdir()
        if isdir(filename)
            @static if RECURSE
                current_directory = pwd()
                cd(filename)
                validate_rktk_files()
                cd(current_directory)
            end
        else
            try
                validate_rktk_file(filename)
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
        end
    end
end


function main()
    if length(ARGS) != 0
        println(stderr, USAGE_STRING)
        exit(EXIT_INVALID_ARGS)
    end
    validate_rktk_files()
end


main()
