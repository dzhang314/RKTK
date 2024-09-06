module RKTKUtilities


using DZOptimization: LBFGSOptimizer, QuadraticLineSearch
using DZOptimization.PCG: random_array
using MultiFloats: MultiFloat, renormalize
using Printf: @sprintf
using RungeKuttaToolKit: RKOCOptimizationProblem


#################################################################### ERROR CODES


export EXIT_MODE_NOT_PROVIDED, EXIT_INVALID_MODE_LENGTH,
    EXIT_INVALID_PARAMETERIZATION, EXIT_INVALID_PRECISION,
    EXIT_INVALID_NUMERIC_LITERAL, EXIT_FILE_DOES_NOT_EXIST,
    EXIT_INPUT_FILE_EMPTY, EXIT_INVALID_PARAMETER_COUNT,
    EXIT_INVALID_ARG_COUNT, EXIT_INVALID_ARG_FORMAT,
    EXIT_INVALID_TREE_ORDERING


const EXIT_MODE_NOT_PROVIDED = 1
const EXIT_INVALID_MODE_LENGTH = 2
const EXIT_INVALID_PARAMETERIZATION = 3
const EXIT_INVALID_PRECISION = 4
const EXIT_INVALID_NUMERIC_LITERAL = 5
const EXIT_FILE_DOES_NOT_EXIST = 6
const EXIT_INPUT_FILE_EMPTY = 7
const EXIT_INVALID_PARAMETER_COUNT = 8
const EXIT_INVALID_ARG_COUNT = 9
const EXIT_INVALID_ARG_FORMAT = 10
const EXIT_INVALID_TREE_ORDERING = 11


################################################################################


export ensuredir


function ensuredir(path::AbstractString)
    if !endswith(pwd(), path)
        if !ispath(path)
            mkpath(path)
        end
        if !isdir(path)
            throw(ArgumentError("Failed to create directory $path."))
        end
        cd(path)
    end
    return nothing
end


################################################################################


export construct_optimizer, reset_occurred


function construct_optimizer(
    seed::UInt64,
    prob::RKOCOptimizationProblem{T},
) where {T}
    n = prob.param.num_variables
    return LBFGSOptimizer(prob, prob', QuadraticLineSearch(),
        random_array(seed, T, n), sqrt(n * eps(T)), n)
end


function reset_occurred(opt::LBFGSOptimizer)
    history_length = length(opt._rho)
    return ((opt.iteration_count[] >= history_length) &&
            (opt._history_count[] != history_length))
end


################################################################################


export uniform_precision_strings


function precision_is_sufficient(x::T, n::Int) where {T}
    s = @sprintf("%+.*e", n, x)
    return parse(T, s) == x
end


function precision_is_sufficient(x::MultiFloat{T,N}, n::Int) where {T,N}
    s = @sprintf("%+.*e", n, x)
    return MultiFloat{T,N}(s) == renormalize(x)
end


precision_is_sufficient(v::AbstractVector{T}, n::Int) where {U} =
    all(precision_is_sufficient(x, n) for x in v)


function find_sufficient_precision(v::AbstractVector{T}) where {T}
    n = precision(T; base=10)
    while true
        if precision_is_sufficient(v, n)
            return n
        end
        n += 1
    end
end


function uniform_precision_strings(v::Vector{U}; sign::Bool=true) where {U}
    n = find_sufficient_precision(v)
    if sign
        return [@sprintf("%+.*e", n, x) for x in v]
    else
        return [@sprintf("%.*e", n, x) for x in v]
    end
end


end # module RKTKUtilities


# using Dates: DateTime, datetime2unix


# const RKTK_SEARCH_FILENAME_REGEX =
#     r"^RKTK-(..)-(..)-(....)-(....)-(....)-(................)\.txt$"
# const RKTK_INCOMPLETE_FILENAME_REGEX =
#     r"^RKTK-([0-9]{2})-([0-9]{2})-(XXXX)-(XXXX)-(XXXX)-([0-9A-F]{16})\.txt$"
# const RKTK_COMPLETE_FILENAME_REGEX =
#     r"^RKTK-([0-9]{2})-([0-9]{2})-([0-9]{4})-([0-9]{4})-([0-9]{4})-([0-9A-F]{16})\.txt$"
# const RKTK_SEARCH_DIRECTORY_REGEX = r"^RKTK-SEARCH-([0-9]{2})-([0-9]{2})$"
# const FILE_CHUNK_SIZE = 4096
# const UNIX_TIME_2024 = datetime2unix(DateTime(2024))


# function read_rktk_search_directory(dirpath::AbstractString)
#     @assert isdir(dirpath)
#     result = Dict{UInt64,String}()
#     orders = Set{Int}()
#     stages = Set{Int}()
#     found = false
#     for filename in readdir(dirpath; sort=false)
#         filepath = abspath(joinpath(dirpath, filename))
#         if isfile(filepath)
#             m = match(RKTK_SEARCH_FILENAME_REGEX, filename)
#             if !isnothing(m)
#                 found = true
#                 @assert mtime(filepath) > UNIX_TIME_2024
#                 m = match(RKTK_COMPLETE_FILENAME_REGEX, filename)
#                 @assert !isnothing(m)
#                 push!(orders, parse(Int, m[1]; base=10))
#                 push!(stages, parse(Int, m[2]; base=10))
#                 id = parse(UInt64, m[6]; base=16)
#                 @assert !haskey(result, id)
#                 result[id] = filename
#             end
#         end
#     end
#     if found
#         @assert isone(length(orders))
#         @assert isone(length(stages))
#     end
#     return result
# end


# function read_rktk_database(dirpath::AbstractString)
#     @assert isdir(dirpath)
#     result = Dict{Tuple{Int,Int},Tuple{String,Dict{UInt64,String}}}()
#     for dirname in readdir(dirpath; sort=false)
#         subpath = abspath(joinpath(dirpath, dirname))
#         if isdir(subpath)
#             m = match(RKTK_SEARCH_DIRECTORY_REGEX, dirname)
#             if !isnothing(m)
#                 order = parse(Int, m[1]; base=10)
#                 stage = parse(Int, m[2]; base=10)
#                 result[(order, stage)] = (subpath,
#                     read_rktk_search_directory(subpath))
#             end
#         end
#     end
#     return result
# end


# function files_are_identical(path1::AbstractString, path2::AbstractString)
#     @assert isfile(path1)
#     @assert isfile(path2)
#     if basename(path1) != basename(path2)
#         return false
#     end
#     return open(path1, "r") do file1
#         return open(path2, "r") do file2
#             seekend(file1)
#             seekend(file2)
#             if position(file1) != position(file2)
#                 return false
#             end
#             seekstart(file1)
#             seekstart(file2)
#             while !(eof(file1) || eof(file2))
#                 chunk1 = read(file1, FILE_CHUNK_SIZE)
#                 chunk2 = read(file2, FILE_CHUNK_SIZE)
#                 if chunk1 != chunk2
#                     return false
#                 end
#             end
#             return eof(file1) && eof(file2)
#         end
#     end
# end


# function count_bits(n::Int)
#     result = 0
#     while !iszero(n)
#         n >>>= 1
#         result += 1
#     end
#     return result
# end


# function find_first_missing_key(dict::Dict{K,V}) where {K,V}
#     for key = typemin(K):typemax(K)
#         if !haskey(dict, key)
#             return key
#         end
#     end
# end


# function clean_floor(n::Int)
#     num_bits = count_bits(n) - 1
#     floor2 = 1 << num_bits
#     floor3 = floor2 | (1 << (num_bits - 1))
#     return ifelse(n >= floor3, floor3, floor2)
# end
