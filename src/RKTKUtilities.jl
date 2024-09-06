module RKTKUtilities


using DZOptimization: LBFGSOptimizer, QuadraticLineSearch
using DZOptimization.Kernels: norm2
using DZOptimization.PCG: random_array
using MultiFloats
using Printf: @sprintf
using RungeKuttaToolKit: RKOCOptimizationProblem
using RungeKuttaToolKit.RKParameterization


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


export is_valid_mode, compute_parameterization


const FLOAT_TYPES = Dict(
    "M1" => Float64,
    "M2" => Float64x2,
    "M3" => Float64x3,
    "M4" => Float64x4,
    "M5" => Float64x5,
    "M6" => Float64x6,
    "M7" => Float64x7,
    "M8" => Float64x8,
    "A1" => BigFloat,
    "A2" => BigFloat,
    "A3" => BigFloat,
    "A4" => BigFloat,
    "A5" => BigFloat,
    "A6" => BigFloat,
    "A7" => BigFloat,
    "A8" => BigFloat,
    "A9" => BigFloat)


const FLOAT_PRECISIONS = Dict(
    "A1" => 512,
    "A2" => 1024,
    "A3" => 2048,
    "A4" => 4096,
    "A5" => 8192,
    "A6" => 16384,
    "A7" => 32768,
    "A8" => 65536,
    "A9" => 131072)


is_valid_mode(mode::AbstractString) =
    (length(mode) == 4) &&
    (mode[1] == 'B') &&
    (mode[2] in ['E', 'D', 'I', 'P', 'Q', 'R', 'S']) &&
    haskey(FLOAT_TYPES, mode[3:4])


function compute_parameterization(mode::AbstractString, stages::Int)
    @assert is_valid_mode(mode)
    if mode[2] == 'E'
        return RKParameterizationExplicit{FLOAT_TYPES[mode[3:4]]}(stages)
    elseif mode[2] == 'D'
        return RKParameterizationDiagonallyImplicit{FLOAT_TYPES[mode[3:4]]}(stages)
    elseif mode[2] == 'I'
        return RKParameterizationImplicit{FLOAT_TYPES[mode[3:4]]}(stages)
    elseif mode[2] == 'P'
        return RKParameterizationParallelExplicit{FLOAT_TYPES[mode[3:4]]}(stages, 2)
    elseif mode[2] == 'Q'
        return RKParameterizationParallelExplicit{FLOAT_TYPES[mode[3:4]]}(stages, 4)
    elseif mode[2] == 'R'
        return RKParameterizationParallelExplicit{FLOAT_TYPES[mode[3:4]]}(stages, 8)
    elseif mode[2] == 'S'
        return RKParameterizationParallelExplicit{FLOAT_TYPES[mode[3:4]]}(stages, 16)
    end
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


export compute_max_residual, compute_rms_gradient, compute_max_coeff,
    compute_residual_score, compute_gradient_score, compute_coeff_score


function compute_max_residual(opt)
    prob = opt.objective_function
    prob.param(prob.A, prob.b, opt.current_point)
    return maximum(abs, prob.ev(prob.A, prob.b))
end


function compute_rms_gradient(opt)
    grad = opt.current_gradient
    return sqrt(norm2(grad) / length(grad))
end


function compute_max_coeff(opt) # TODO: QR
    prob = opt.objective_function
    prob.param(prob.A, prob.b, opt.current_point)
    return max(maximum(abs, prob.A), maximum(abs, prob.b))
end


compute_residual_score(opt) = round(Int, clamp(
    -500 * log10(Float64(compute_max_residual(opt))),
    0.0, 9999.0))


compute_gradient_score(opt) = round(Int, clamp(
    -500 * log10(Float64(compute_rms_gradient(opt))),
    0.0, 9999.0))


compute_coeff_score(opt) = round(Int, clamp(
    10000 - 2500 * log10(Float64(compute_max_coeff(opt))),
    0.0, 9999.0))


################################################################################


export uniform_precision_strings


function precision_is_sufficient(x::T, n::Int) where {T}
    s = @sprintf("%+.*e", n, x)
    return parse(T, s) == x
end


function precision_is_sufficient(x::MultiFloat{T,N}, n::Int) where {T,N}
    s = @sprintf("%+.*e", n, x)
    return MultiFloat{T,N}(s) == MultiFloats.renormalize(x)
end


precision_is_sufficient(v::AbstractVector, n::Int) =
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


function uniform_precision_strings(v::AbstractVector; sign::Bool=true)
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
