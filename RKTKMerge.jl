using Base.Threads
using Printf


push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using RKTKUtilities


const USAGE_STRING = """
Usage: julia [jl_options] $PROGRAM_FILE
[jl_options] refers to Julia options, such as -O3.

Merge all RKTK files in the current directory into the RKTK database whose
location is specified in the RKTK_DATABASE_DIRECTORY environment variable.
If duplicates are found, their contents are checked, and if they match, the
older file is kept and the newer file is deleted.

This program does not perform any validation of the contents of the files
being merged. Use RKTKValidate.jl before or after merging for this purpose.

Options:
    --recurse           Recursively merge RKTK files in subdirectories.
    --verbose           Print names of successfully merged RKTK files.
"""


const RKTK_DATABASE_DOES_NOT_EXIST_STRING = """
ERROR: Before running $PROGRAM_FILE, you must set the environment variable
RKTK_DATABASE_DIRECTORY to the absolute path of an existing directory.
"""


const INSIDE_RKTK_DATABASE_STRING = """
ERROR: $PROGRAM_FILE cannot be run from inside the RKTK database.
"""


const SINGLE_THREADED_STRING = """
WARNING: Multi-threaded execution was requested, but $PROGRAM_FILE is a
single-threaded program. Execution will continue using only one thread.
"""


const RECURSE = get_flag!(["recurse"])
const VERBOSE = get_flag!(["verbose"])


function get_rktk_database_directory()
    if haskey(ENV, "RKTK_DATABASE_DIRECTORY")
        envdir = ENV["RKTK_DATABASE_DIRECTORY"]
        if isdir(envdir)
            return realpath(abspath(envdir))
        end
    end
    if haskey(ENV, "HOME")
        homedir = joinpath(ENV["HOME"], "RKTK-DATABASE")
        if isdir(homedir)
            return realpath(abspath(envdir))
        end
    end
    return nothing
end


const RKTK_DATABASE_DIRECTORY = get_rktk_database_directory()
if (isnothing(RKTK_DATABASE_DIRECTORY) ||
    !isdir(RKTK_DATABASE_DIRECTORY) ||
    !isabspath(RKTK_DATABASE_DIRECTORY))
    print(stderr, RKTK_DATABASE_DOES_NOT_EXIST_STRING)
    exit(EXIT_RKTK_DATABASE_DOES_NOT_EXIST)
end


if startswith(realpath(pwd()), RKTK_DATABASE_DIRECTORY)
    print(stderr, INSIDE_RKTK_DATABASE_STRING)
    exit(EXIT_INSIDE_RKTK_DATABASE)
end


if nthreads() > 1
    print(stderr, SINGLE_THREADED_STRING)
end


const RKTK_DATABASE_DIRS = Dict{String,Dict{UInt64,String}}()
for dirname in readdir(RKTK_DATABASE_DIRECTORY)
    @assert !isnothing(match(RKTK_DIRECTORY_REGEX, dirname))
    dirpath = joinpath(RKTK_DATABASE_DIRECTORY, dirname)
    @assert isabspath(dirpath) && isdir(dirpath)
    dirdict = Dict{UInt64,String}()
    for filename in readdir(dirpath)
        filepath = joinpath(dirpath, filename)
        m = match(RKTK_FILENAME_REGEX, filename)
        @assert !isnothing(m)
        seed = parse(UInt64, m[7]; base=16)
        @assert !haskey(dirdict, seed)
        dirdict[seed] = filename
    end
    RKTK_DATABASE_DIRS[dirname] = dirdict
end


function add_file_to_database!(m::RegexMatch)
    @assert m.regex == RKTK_FILENAME_REGEX
    filename = m.match
    @assert isfile(filename)
    dirname = @sprintf("RKTK-%s-%s-%s", m[1], m[2], m[3][1:2])
    dirpath = joinpath(RKTK_DATABASE_DIRECTORY, dirname)
    if !haskey(RKTK_DATABASE_DIRS, dirname)
        @assert !ispath(dirpath)
        mkdir(dirpath)
        RKTK_DATABASE_DIRS[dirname] = Dict{UInt64,String}()
    end
    @assert isdir(dirpath)
    dirdict = RKTK_DATABASE_DIRS[dirname]
    seed = parse(UInt64, m[7]; base=16)
    if haskey(dirdict, seed)
        refpath = joinpath(dirpath, dirdict[seed])
        @assert files_are_identical(refpath, filename)
        reftime = mtime(refpath)
        @assert reftime > UNIX_TIME_2024
        filetime = mtime(filename)
        @assert filetime > UNIX_TIME_2024
        if filetime < reftime
            rm(refpath)
            mv(filename, refpath)
            @static if VERBOSE
                println("Replaced existing file in RKTK database: $filename")
            end
        else
            rm(filename)
            @static if VERBOSE
                println("Deleted identical duplicate RKTK file: $filename")
            end
        end
    else
        dirdict[seed] = filename
        mv(filename, joinpath(dirpath, filename))
        @static if VERBOSE
            println("Added new file to RKTK database: $filename")
        end
    end
    return nothing
end


const MERGED_FILE_COUNTER = Atomic{UInt64}(0)


function try_readdir(dirname::AbstractString)
    try
        result = readdir(dirname)
        println("Scanning directory: $dirname")
        return result
    catch e
        if e isa Base.IOError
            println("Skipping unreadable directory: $dirname")
            return String[]
        else
            rethrow(e)
        end
    end
end


function add_directory_to_database!()
    realpwd = realpath(pwd())
    if startswith(realpwd, RKTK_DATABASE_DIRECTORY)
        return nothing
    end
    for filename in try_readdir(realpwd)
        if isdir(filename)
            @static if RECURSE
                cd(filename)
                add_directory_to_database!()
                cd(realpwd)
            end
        else
            m = match(RKTK_FILENAME_REGEX, filename)
            if !isnothing(m)
                add_file_to_database!(m)
                atomic_add!(MERGED_FILE_COUNTER, one(UInt64))
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
    add_directory_to_database!()
    end_time = time_ns()
    elapsed_time = (end_time - start_time) / 1.0e9
    @printf("Successfully merged %d RKTK files in %g seconds (%g files per second).\n",
        MERGED_FILE_COUNTER[], elapsed_time, MERGED_FILE_COUNTER[] / elapsed_time)
    return nothing
end


main()
