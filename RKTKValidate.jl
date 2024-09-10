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
        print(stderr, USAGE_STRING)
        exit(EXIT_INVALID_ARGS)
    end
    start_time = time_ns()
    process_directory()
    end_time = time_ns()
    elapsed_time = (end_time - start_time) / 1.0e9
    @printf("Successfully validated %d RKTK files in %g seconds (%g files per second).\n",
        VALID_FILE_COUNTER[], elapsed_time, VALID_FILE_COUNTER[] / elapsed_time)
    return nothing
end


main()
