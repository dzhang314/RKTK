module RKTKUtilities


using Base.Unicode: category_code, UTF8PROC_CATEGORY_PC, UTF8PROC_CATEGORY_PD
using DZOptimization: LBFGSOptimizer, QuadraticLineSearch
using DZOptimization.Kernels: norm2
using DZOptimization.PCG: random_array
using MultiFloats
using Printf: @sprintf
using RungeKuttaToolKit: RKOCOptimizationProblem
using RungeKuttaToolKit.RKParameterization
using Unicode: isequal_normalized


#################################################################### ERROR CODES


export EXIT_INVALID_ARGS


const EXIT_INVALID_ARGS = 1


################################################################################


export get_flag!


@inline function underscore_chartransform(c::Integer)
    cat = category_code(c)
    return ((cat == UTF8PROC_CATEGORY_PC) || (cat == UTF8PROC_CATEGORY_PD) ?
            convert(typeof(c), '-') : c)
end


@inline insensitive_match(s::AbstractString, t::AbstractString) =
    isequal_normalized(s, t; casefold=true, stripmark=true,
        chartransform=underscore_chartransform)


is_arg(s, names) = startswith(s, "--") && any(
    insensitive_match(s[3:end], name) for name in names)


function get_flag!(names)
    matches = [is_arg(arg, names) for arg in ARGS]
    deleteat!(ARGS, matches)
    return any(matches)
end


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


export is_valid_mode, get_type, get_parameterization


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


function get_type(mode::AbstractString)
    result = FLOAT_TYPES[mode[3:4]]
    if result == BigFloat
        setprecision(BigFloat, FLOAT_PRECISIONS[mode[3:4]])
    end
    return result
end


function get_parameterization(mode::AbstractString, stages::Int)
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


function construct_optimizer(
    prob::RKOCOptimizationProblem{T},
    x::AbstractVector{T},
) where {T}
    n = length(x)
    @assert n == prob.param.num_variables
    return LBFGSOptimizer(prob, prob', QuadraticLineSearch(),
        x, sqrt(n * eps(T)), n)
end


function reset_occurred(opt::LBFGSOptimizer)
    history_length = length(opt._rho)
    return ((opt.iteration_count[] >= history_length) &&
            (opt._history_count[] != history_length))
end


################################################################################


export compute_max_residual, compute_rms_gradient, compute_max_coeff,
    compute_residual_score, compute_gradient_score, compute_coeff_score,
    TABLE_HEADER, TABLE_SEPARATOR, compute_table_row


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


const TABLE_HEADER = "| ITERATION |  COST FUNC.  | MAX RESIDUAL | RMS GRADIENT |  MAX COEFF.  |  STEPLENGTH  |"
const TABLE_SEPARATOR = "|-----------|--------------|--------------|--------------|--------------|--------------|"


compute_table_row(opt) = @sprintf(
    "|%10d | %.6e | %.6e | %.6e | %.6e | %.6e |%s",
    opt.iteration_count[],
    opt.current_objective_value[],
    compute_max_residual(opt),
    compute_rms_gradient(opt),
    compute_max_coeff(opt),
    sqrt(norm2(opt.delta_point)),
    reset_occurred(opt) ? " RESET" : "")


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


################################################################################


export RKTK_FILENAME_REGEX, RKTK_INCOMPLETE_FILENAME_REGEX


const RKTK_FILENAME_REGEX =
    r"^RKTK-([0-9]{2})-([0-9]{2})-([0-9A-Za-z]{4})-([0-9]{4}|FAIL)-([0-9]{4}|FAIL)-([0-9]{4}|FAIL)-([0-9A-Fa-f]{16}).txt$"
const RKTK_INCOMPLETE_FILENAME_REGEX =
    r"^RKTK-([0-9]{2})-([0-9]{2})-([0-9A-Za-z]{4})-XXXX-XXXX-XXXX-([0-9A-Fa-f]{16}).txt$"


################################################################################


export parse_floats, ParsedRKTKTableRow, full_match, partial_match


function parse_floats(::Type{T}, data::AbstractString) where {T}
    return parse.(T, split(strip(data, '\n'), '\n'))
end


function parse_floats(::Type{MultiFloat{T,N}}, data::AbstractString) where {T,N}
    return MultiFloat{T,N}.(split(strip(data, '\n'), '\n'))
end


struct ParsedRKTKTableRow
    iteration::Int
    cost_func::BigFloat
    max_residual::BigFloat
    rms_gradient::BigFloat
    max_coeff::BigFloat
    steplength::BigFloat

    function ParsedRKTKTableRow(row::AbstractString)
        @assert count('|', row) == 7
        entries = split(row, '|')
        @assert isempty(entries[1])
        iteration = parse(Int, strip(entries[2]))
        cost_func = parse(BigFloat, strip(entries[3]))
        max_residual = parse(BigFloat, strip(entries[4]))
        rms_gradient = parse(BigFloat, strip(entries[5]))
        max_coeff = parse(BigFloat, strip(entries[6]))
        steplength = parse(BigFloat, strip(entries[7]))
        return new(iteration, cost_func, max_residual,
            rms_gradient, max_coeff, steplength)
    end
end


function full_match(r::ParsedRKTKTableRow, s::ParsedRKTKTableRow)
    return ((r.iteration == s.iteration) &&
            (r.cost_func == s.cost_func) &&
            (r.max_residual == s.max_residual) &&
            (r.rms_gradient == s.rms_gradient) &&
            (r.max_coeff == s.max_coeff) &&
            (r.steplength == s.steplength))
end


function partial_match(r::ParsedRKTKTableRow, s::ParsedRKTKTableRow)
    return ((r.cost_func == s.cost_func) &&
            (r.max_residual == s.max_residual) &&
            (r.rms_gradient == s.rms_gradient) &&
            (r.max_coeff == s.max_coeff))
end


end # module RKTKUtilities


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


# function clean_floor(n::Int)
#     num_bits = count_bits(n) - 1
#     floor2 = 1 << num_bits
#     floor3 = floor2 | (1 << (num_bits - 1))
#     return ifelse(n >= floor3, floor3, floor2)
# end
