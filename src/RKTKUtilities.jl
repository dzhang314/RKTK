module RKTKUtilities


using Base.Unicode: category_code, UTF8PROC_CATEGORY_PC, UTF8PROC_CATEGORY_PD
using Dates: DateTime, datetime2unix
using DZOptimization: LBFGSOptimizer, QuadraticLineSearch
using DZOptimization.Kernels: norm2
using DZOptimization.PCG: random_array
using MultiFloats
using Printf
using RungeKuttaToolKit
using RungeKuttaToolKit.RKCost
using RungeKuttaToolKit.RKParameterization
using Unicode: isequal_normalized


#################################################################### ERROR CODES


export EXIT_INVALID_ARGS, EXIT_RKTK_DATABASE_DOES_NOT_EXIST,
    EXIT_INSIDE_RKTK_DATABASE


const EXIT_INVALID_ARGS = 1
const EXIT_RKTK_DATABASE_DOES_NOT_EXIST = 2
const EXIT_INSIDE_RKTK_DATABASE = 3


############################################################### ARGUMENT PARSING


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


########################################################### FILESYSTEM UTILITIES


export UNIX_TIME_2024, ensuredir, files_are_identical


const UNIX_TIME_2024 = datetime2unix(DateTime(2024))


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


const FILE_CHUNK_SIZE = 67108864 # 64 MiB


function files_are_identical(p::AbstractString, q::AbstractString)
    @assert isfile(p)
    @assert isfile(q)
    if basename(p) != basename(q)
        return false
    end
    return open(p, "r") do f
        return open(q, "r") do g
            seekend(f)
            seekend(g)
            if position(f) != position(g)
                return false
            end
            seekstart(f)
            seekstart(g)
            while !(eof(f) || eof(g))
                if read(f, FILE_CHUNK_SIZE) != read(g, FILE_CHUNK_SIZE)
                    return false
                end
            end
            return eof(f) && eof(g)
        end
    end
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
    "A0" => BigFloat,
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
    "A0" => 256,
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
        return RKParameterizationExplicit{get_type(mode)}(stages)
    elseif mode[2] == 'D'
        return RKParameterizationDiagonallyImplicit{get_type(mode)}(stages)
    elseif mode[2] == 'I'
        return RKParameterizationImplicit{get_type(mode)}(stages)
    elseif mode[2] == 'P'
        return RKParameterizationParallelExplicit{get_type(mode)}(stages, 2)
    elseif mode[2] == 'Q'
        return RKParameterizationParallelExplicit{get_type(mode)}(stages, 4)
    elseif mode[2] == 'R'
        return RKParameterizationParallelExplicit{get_type(mode)}(stages, 8)
    elseif mode[2] == 'S'
        return RKParameterizationParallelExplicit{get_type(mode)}(stages, 16)
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


function construct_optimizer(
    trees::AbstractVector{LevelSequence},
    param::AbstractRKParameterization{T},
    x::AbstractVector{T},
) where {T}
    n = length(x)
    @assert n == param.num_variables
    ev = RKOCEvaluator{T}(trees, param.num_stages)
    prob = RKOCOptimizationProblem(ev, RKCostL2{T}(), param)
    return construct_optimizer(prob, x)
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


export RKTK_FILENAME_REGEX, RKTK_INCOMPLETE_FILENAME_REGEX,
    RKTK_DIRECTORY_REGEX


const RKTK_FILENAME_REGEX =
    r"^RKTK-([0-9]{2})-([0-9]{2})-([0-9A-Za-z]{4})-([0-9]{4}|FAIL)-([0-9]{4}|FAIL)-([0-9]{4}|FAIL)-([0-9A-Fa-f]{16}).txt$"
const RKTK_INCOMPLETE_FILENAME_REGEX =
    r"^RKTK-([0-9]{2})-([0-9]{2})-([0-9A-Za-z]{4})-XXXX-XXXX-XXXX-([0-9A-Fa-f]{16}).txt$"
const RKTK_DIRECTORY_REGEX =
    r"^RKTK-([0-9]{2})-([0-9]{2})-([0-9A-Za-z]{2})$"


################################################################################


export parse_floats, assert_rktk_file_valid


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


function assert_rktk_file_valid(m::RegexMatch)
    @assert m.regex == RKTK_FILENAME_REGEX

    order = parse(Int, m[1]; base=10)
    stages = parse(Int, m[2]; base=10)
    mode = m[3]
    residual_score = parse(Int, m[4]; base=10)
    gradient_score = parse(Int, m[5]; base=10)
    coeff_score = m[6] == "FAIL" ? -1 : parse(Int, m[6])
    seed = parse(UInt64, m[7]; base=16)

    T = get_type(mode)
    param = get_parameterization(mode, stages)
    prob = RKOCOptimizationProblem(
        RKOCEvaluator{T}(order, param.num_stages),
        RKCostL2{T}(), param)

    blocks = split(read(m.match, String), "\n\n")
    @assert length(blocks) == 3
    @assert !startswith(blocks[1], '\n')
    @assert !endswith(blocks[1], '\n')
    @assert !startswith(blocks[2], '\n')
    @assert !endswith(blocks[2], '\n')
    @assert !startswith(blocks[3], '\n')
    @assert endswith(blocks[3], '\n')

    initial = parse_floats(T, blocks[1])
    @assert initial == random_array(seed, T, param.num_variables)

    table = split(strip(blocks[2], '\n'), '\n')
    @assert length(table) >= 4
    @assert table[1] == TABLE_HEADER
    @assert table[2] == TABLE_SEPARATOR

    table = ParsedRKTKTableRow.(table[3:end])
    @assert issorted(row.iteration for row in table)
    @assert issorted(row.cost_func for row in table; rev=true)
    @assert all(!signbit(row.max_residual) for row in table)
    @assert all(!signbit(row.rms_gradient) for row in table)
    @assert all(!signbit(row.max_coeff) for row in table)
    @assert all(!signbit(row.steplength) for row in table)

    final = parse_floats(T, blocks[3])
    @assert length(final) == param.num_variables

    opt_initial = construct_optimizer(prob, initial)
    initial_row = ParsedRKTKTableRow(compute_table_row(opt_initial))
    opt_final = construct_optimizer(prob, final)
    final_row = ParsedRKTKTableRow(compute_table_row(opt_final))
    @assert full_match(initial_row, table[1])
    @assert partial_match(final_row, table[end])

    @assert compute_residual_score(opt_final) == residual_score
    @assert compute_gradient_score(opt_final) == gradient_score
    if coeff_score == -1
        @assert compute_coeff_score(opt_final) <= 5000
    else
        @assert compute_coeff_score(opt_final) == coeff_score
    end

    return nothing
end


end # module RKTKUtilities
