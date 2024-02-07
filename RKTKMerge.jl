include("RKTKUtilities.jl")


const RKTK_DATABASE_DIRECTORY =
    haskey(ENV, "RKTK_DATABASE_DIRECTORY") ?
    normpath(ENV["RKTK_DATABASE_DIRECTORY"]) :
    abspath(joinpath(ENV["HOME"], "RKTK-DATABASE"))
@assert isabspath(RKTK_DATABASE_DIRECTORY)
@assert isdir(RKTK_DATABASE_DIRECTORY)


const RKTK_DATABASE = read_rktk_database(RKTK_DATABASE_DIRECTORY)


function merge(dirpath::AbstractString)
    @assert isdir(dirpath)
    println("Merging $dirpath into RKTK database...")
    num_added = 0
    num_deleted = 0
    for filename in readdir(dirpath; sort=false)
        filepath = abspath(joinpath(dirpath, filename))
        if isfile(filepath)
            m = match(RKTK_SEARCH_FILENAME_REGEX, filename)
            if !isnothing(m)
                incomplete_match = match(
                    RKTK_INCOMPLETE_FILENAME_REGEX, filename)
                is_incomplete = !isnothing(incomplete_match)
                complete_match = match(
                    RKTK_COMPLETE_FILENAME_REGEX, filename)
                is_complete = !isnothing(complete_match)
                @assert xor(is_incomplete, is_complete)
                if is_complete
                    @assert mtime(filepath) > UNIX_TIME_2024
                    order = parse(Int, complete_match[1]; base=10)
                    stage = parse(Int, complete_match[2]; base=10)
                    id = parse(UInt64, complete_match[6]; base=16)
                    if !haskey(RKTK_DATABASE, (order, stage))
                        newpath = abspath(joinpath(RKTK_DATABASE_DIRECTORY,
                            @sprintf("RKTK-SEARCH-%02d-%02d", order, stage)))
                        mkdir(newpath)
                        RKTK_DATABASE[(order, stage)] =
                            (newpath, Dict{UInt64,String}())
                    end
                    dbdir, db = RKTK_DATABASE[(order, stage)]
                    if haskey(db, id)
                        dbpath = abspath(joinpath(dbdir, db[id]))
                        @assert files_are_identical(dbpath, filepath)
                        if mtime(filepath) < mtime(dbpath)
                            rm(dbpath)
                            mv(filepath, dbpath)
                        else
                            rm(filepath)
                        end
                        @assert isfile(dbpath)
                        @assert !isfile(filepath)
                        num_deleted += 1
                    else
                        dbpath = abspath(joinpath(dbdir, filename))
                        mv(filepath, dbpath)
                        db[id] = dbpath
                        @assert isfile(dbpath)
                        @assert !isfile(filepath)
                        num_added += 1
                    end
                end
            end
        end
    end
    println("Added $num_added new files.")
    println("Deleted $num_deleted duplicate files.")
end


function main()
    for dirpath in ARGS
        merge(dirpath)
    end
end


main()
