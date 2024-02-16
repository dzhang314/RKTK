include("RKTKUtilities.jl")
using Printf: @sprintf


const RKTK_DATABASE_DIRECTORY =
    haskey(ENV, "RKTK_DATABASE_DIRECTORY") ?
    normpath(ENV["RKTK_DATABASE_DIRECTORY"]) :
    abspath(joinpath(ENV["HOME"], "RKTK-DATABASE"))
@assert isabspath(RKTK_DATABASE_DIRECTORY)
@assert isdir(RKTK_DATABASE_DIRECTORY)


const RKTK_DATABASE = read_rktk_database(RKTK_DATABASE_DIRECTORY)


function main()
    for ((order, num_stages), (dirpath, db)) in RKTK_DATABASE
        clean_count = clean_floor(Int(find_first_missing_key(db)))
        println("Freezing ", clean_count,
            " entries from directory ", dirpath, "...")
        freezedir = @sprintf("RKTK-SEARCH-%02d-%02d-%016X",
            order, num_stages, clean_count - 1)
        mkdir(freezedir)
        for key = UInt64(0):UInt64(clean_count - 1)
            cp(joinpath(dirpath, db[key]), joinpath(freezedir, db[key]))
        end
    end
end


main()
