const RKTK_FILENAME_REGEX =
    r"^RKTK-..-..-....-....-....-................\.txt$"
const RKTK_INCOMPLETE_REGEX =
    r"^RKTK-([0-9]{2})-([0-9]{2})-(XXXX)-(XXXX)-(XXXX)-([0-9A-F]{16})\.txt$"
const RKTK_COMPLETE_REGEX =
    r"^RKTK-([0-9]{2})-([0-9]{2})-([0-9]{4})-([0-9]{4})-([0-9]{4})-([0-9A-F]{16})\.txt$"
const FILE_CHUNK_SIZE = 4096


function read_rktk_search_directory(dirpath::AbstractString)
    filenames = filter!(
        name -> !isnothing(match(RKTK_FILENAME_REGEX, name)),
        readdir(dirpath; sort=false))
    result = Dict{UInt64,String}()
    if iszero(length(filenames))
        return result
    end
    orders = Set{Int}()
    stages = Set{Int}()
    for filename in filenames
        incomplete_match = match(RKTK_INCOMPLETE_REGEX, filename)
        is_incomplete = !isnothing(incomplete_match)
        if is_incomplete
            push!(orders, parse(Int, incomplete_match[1]; base=10))
            push!(stages, parse(Int, incomplete_match[2]; base=10))
            id = parse(UInt64, incomplete_match[6]; base=16)
            @assert !haskey(result, id)
            result[id] = filename
        end
        complete_match = match(RKTK_COMPLETE_REGEX, filename)
        is_complete = !isnothing(complete_match)
        if is_complete
            push!(orders, parse(Int, complete_match[1]; base=10))
            push!(stages, parse(Int, complete_match[2]; base=10))
            id = parse(UInt64, complete_match[6]; base=16)
            @assert !haskey(result, id)
            result[id] = filename
        end
        @assert xor(is_incomplete, is_complete)
    end
    @assert isone(length(orders))
    @assert isone(length(stages))
    return result
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
