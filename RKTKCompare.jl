const RKTK_FILENAME_REGEX = r"^RKTK-..-..-....-....-....-................\.txt$"
const RKTK_INCOMPLETE_REGEX = r"^RKTK-([0-9]{2})-([0-9]{2})-(XXXX)-(XXXX)-(XXXX)-([0-9A-F]{16})\.txt$"
const RKTK_COMPLETE_REGEX = r"^RKTK-([0-9]{2})-([0-9]{2})-([0-9]{4})-([0-9]{4})-([0-9]{4})-([0-9A-F]{16})\.txt$"
const FILE_CHUNK_SIZE = 1024


function read_rktk_search_directory(dirpath::AbstractString)
    filenames = readdir(dirpath; sort=false)
    filter!(name -> !isnothing(match(RKTK_FILENAME_REGEX, name)), filenames)
    complete_matches = RegexMatch[]
    if iszero(length(filenames))
        return complete_matches
    end
    orders = Set{Int}()
    stages = Set{Int}()
    ids = Set{UInt64}()
    for filename in filenames
        incomplete_match = match(RKTK_INCOMPLETE_REGEX, filename)
        if !isnothing(incomplete_match)
            push!(orders, parse(Int, incomplete_match[1]; base=10))
            push!(stages, parse(Int, incomplete_match[2]; base=10))
            push!(ids, parse(UInt64, incomplete_match[6]; base=16))
        end
        complete_match = match(RKTK_COMPLETE_REGEX, filename)
        if !isnothing(complete_match)
            push!(orders, parse(Int, complete_match[1]; base=10))
            push!(stages, parse(Int, complete_match[2]; base=10))
            push!(ids, parse(UInt64, complete_match[6]; base=16))
        end
        @assert xor(isnothing(incomplete_match), isnothing(complete_match))
        if !isnothing(complete_match)
            push!(complete_matches, complete_match)
        end
    end
    @assert isone(length(orders))
    @assert isone(length(stages))
    @assert length(ids) == length(filenames)
    return complete_matches
end


function files_are_identical(path1::AbstractString, path2::AbstractString)
    if basename(path1) != basename(path2)
        return false
    end
    open(path1, "r") do file1
        open(path2, "r") do file2
            seekend(file1)
            seekend(file2)
            if position(file1) != position(file2)
                return false
            end
            seekstart(file1)
            seekstart(file2)
            while !eof(file1) && !eof(file2)
                chunk1 = read(file1, FILE_CHUNK_SIZE)
                chunk2 = read(file2, FILE_CHUNK_SIZE)
                if chunk1 != chunk2
                    return false
                end
            end
            return eof(file1) && eof(file2)
        end
    end
end


function compare_rktk_search_directories(
    dirpath1::AbstractString, dirpath2::AbstractString
)
    println("Scanning:")
    println("    $dirpath1")
    println("    $dirpath2")
    entries1 = Dict(parse(UInt64, m[6]; base=16) => m.match
                    for m in read_rktk_search_directory(dirpath1))
    entries2 = Dict(parse(UInt64, m[6]; base=16) => m.match
                    for m in read_rktk_search_directory(dirpath2))
    common_keys = intersect(keys(entries1), keys(entries2))
    num_keys = length(common_keys)
    println("These directories have $num_keys common files.")

    error_found = false
    for key in common_keys
        filename1 = entries1[key]
        filename2 = entries2[key]
        filepath1 = joinpath(dirpath1, filename1)
        filepath2 = joinpath(dirpath2, filename2)
        if !files_are_identical(filepath1, filepath2)
            println("ERROR: $filepath1 has different" *
                    " contents from $filepath2.")
            error_found = true
        end
    end
    if !error_found
        println("All common files are identical.")
    end

    diff1 = length(setdiff(keys(entries1), keys(entries2)))
    if iszero(diff1)
        println("All files in the first directory" *
                " are present in the second.")
    else
        println("The first directory contains " *
                "$diff1 files that the second does not.")
    end

    diff2 = length(setdiff(keys(entries2), keys(entries1)))
    if iszero(diff2)
        println("All files in the second directory" *
                " are present in the first.")
    else
        println("The second directory contains " *
                "$diff2 files that the first does not.")
    end

    println()
end


function main()
    if length(ARGS) == 2
        compare_rktk_search_directories(ARGS[1], ARGS[2])
    else
        compare_rktk_search_directories(
            "/home/dkzhang/RKTK-DATABASE/RKTK-SEARCH-08-10-0000000000001FFF",
            "/home/dkzhang/RKTK-COMPARE/RKTK-SEARCH-08-10")
        compare_rktk_search_directories(
            "/home/dkzhang/RKTK-DATABASE/RKTK-SEARCH-08-11-0000000000000FFF",
            "/home/dkzhang/RKTK-COMPARE/RKTK-SEARCH-08-11")
        compare_rktk_search_directories(
            "/home/dkzhang/RKTK-DATABASE/RKTK-SEARCH-10-16-0000000000000FFF",
            "/home/dkzhang/RKTK-COMPARE/RKTK-SEARCH-10-16")
    end
end


main()
