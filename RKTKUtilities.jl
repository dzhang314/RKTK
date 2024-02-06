const RKTK_SEARCH_FILENAME_REGEX =
    r"^RKTK-(..)-(..)-(....)-(....)-(....)-(................)\.txt$"
const RKTK_INCOMPLETE_FILENAME_REGEX =
    r"^RKTK-([0-9]{2})-([0-9]{2})-(XXXX)-(XXXX)-(XXXX)-([0-9A-F]{16})\.txt$"
const RKTK_COMPLETE_FILENAME_REGEX =
    r"^RKTK-([0-9]{2})-([0-9]{2})-([0-9]{4})-([0-9]{4})-([0-9]{4})-([0-9A-F]{16})\.txt$"
const RKTK_SEARCH_DIRECTORY_REGEX = r"^RKTK-SEARCH-([0-9]{2})-([0-9]{2})$"
const FILE_CHUNK_SIZE = 4096


function read_rktk_search_directory(dirpath::AbstractString)
    @assert isdir(dirpath)
    filenames = filter!(
        name -> !isnothing(match(RKTK_SEARCH_FILENAME_REGEX, name)),
        readdir(dirpath; sort=false))
    result = Dict{UInt64,String}()
    if iszero(length(filenames))
        return result
    end
    orders = Set{Int}()
    stages = Set{Int}()
    for filename in filenames
        complete_match = match(RKTK_COMPLETE_FILENAME_REGEX, filename)
        @assert !isnothing(complete_match)
        push!(orders, parse(Int, complete_match[1]; base=10))
        push!(stages, parse(Int, complete_match[2]; base=10))
        id = parse(UInt64, complete_match[6]; base=16)
        @assert !haskey(result, id)
        result[id] = filename
    end
    @assert isone(length(orders))
    @assert isone(length(stages))
    return result
end


function read_rktk_database(dirpath::AbstractString)
    @assert isdir(dirpath)
    result = Dict{Tuple{Int,Int},Tuple{String,Dict{UInt64,String}}}()
    for dirname in readdir(dirpath)
        subpath = abspath(joinpath(dirpath, dirname))
        if isdir(subpath)
            m = match(RKTK_SEARCH_DIRECTORY_REGEX, dirname)
            if !isnothing(m)
                order = parse(Int, m[1]; base=10)
                num_stages = parse(Int, m[2]; base=10)
                result[(order, num_stages)] = (subpath,
                    read_rktk_search_directory(subpath))
            end
        end
    end
    return result
end


function files_are_identical(path1::AbstractString, path2::AbstractString)
    if basename(path1) != basename(path2)
        return false
    end
    return open(path1, "r") do file1
        return open(path2, "r") do file2
            seekend(file1)
            seekend(file2)
            if position(file1) != position(file2)
                return false
            end
            seekstart(file1)
            seekstart(file2)
            while !(eof(file1) || eof(file2))
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


function count_bits(n::Int)
    result = 0
    while !iszero(n)
        n >>>= 1
        result += 1
    end
    return result
end


function clean_floor(n::Int)
    num_bits = count_bits(n) - 1
    floor2 = 1 << num_bits
    floor3 = floor2 | (1 << (num_bits - 1))
    return ifelse(n >= floor3, floor3, floor2)
end
