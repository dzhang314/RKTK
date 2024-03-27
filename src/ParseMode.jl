# WARNING: This file should not be executed directly. It should be included in
# other RKTK scripts using the include() function in order to parse the first
# command-line argument, which specifies the RKTK operating mode.


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


if isempty(ARGS)
    println(stderr, "ERROR: No mode string provided.")
    print(stderr, USAGE_STRING)
    exit(EXIT_MODE_NOT_PROVIDED)
end


function parse_mode(mode::String)
    if length(mode) != 4
        println(stderr, "ERROR: Invalid mode string.")
        print(stderr, USAGE_STRING)
        exit(EXIT_INVALID_MODE_LENGTH)
    end
    if !(mode[1] in ['A', 'B'])
        println(stderr, "ERROR: Invalid mode string.")
        print(stderr, USAGE_STRING)
        exit(EXIT_INVALID_PARAMETERIZATION)
    end
    if !(mode[2] in ['E', 'D', 'I'])
        println(stderr, "ERROR: Invalid mode string.")
        print(stderr, USAGE_STRING)
        exit(EXIT_INVALID_PARAMETERIZATION)
    end
    if !(mode[3:4] in ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8",
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"])
        println(stderr, "ERROR: Invalid mode string.")
        print(stderr, USAGE_STRING)
        exit(EXIT_INVALID_PRECISION)
    end
    return (Symbol(mode[1:2]), Symbol(mode[3:4]))
end


const PARAMETERIZATION, PRECISION = parse_mode(ARGS[1])


@static if PRECISION == :M1
    const T = Float64
    const PREV_PRECISION = nothing
    const NEXT_PRECISION = :M2
elseif PRECISION == :M2
    const T = Float64x2
    const PREV_PRECISION = :M1
    const NEXT_PRECISION = :M3
elseif PRECISION == :M3
    const T = Float64x3
    const PREV_PRECISION = :M2
    const NEXT_PRECISION = :M4
elseif PRECISION == :M4
    const T = Float64x4
    const PREV_PRECISION = :M3
    const NEXT_PRECISION = :M5
elseif PRECISION == :M5
    const T = Float64x5
    const PREV_PRECISION = :M4
    const NEXT_PRECISION = :M6
elseif PRECISION == :M6
    const T = Float64x6
    const PREV_PRECISION = :M5
    const NEXT_PRECISION = :M7
elseif PRECISION == :M7
    const T = Float64x7
    const PREV_PRECISION = :M6
    const NEXT_PRECISION = :M8
elseif PRECISION == :M8
    const T = Float64x8
    const PREV_PRECISION = :M7
    const NEXT_PRECISION = :A1
elseif PRECISION == :A1
    setprecision(512)
    const T = BigFloat
    const PREV_PRECISION = :M8
    const NEXT_PRECISION = :A2
elseif PRECISION == :A2
    setprecision(1024)
    const T = BigFloat
    const PREV_PRECISION = :A1
    const NEXT_PRECISION = :A3
elseif PRECISION == :A3
    setprecision(2048)
    const T = BigFloat
    const PREV_PRECISION = :A2
    const NEXT_PRECISION = :A4
elseif PRECISION == :A4
    setprecision(4096)
    const T = BigFloat
    const PREV_PRECISION = :A3
    const NEXT_PRECISION = :A5
elseif PRECISION == :A5
    setprecision(8192)
    const T = BigFloat
    const PREV_PRECISION = :A4
    const NEXT_PRECISION = :A6
elseif PRECISION == :A6
    setprecision(16384)
    const T = BigFloat
    const PREV_PRECISION = :A5
    const NEXT_PRECISION = :A7
elseif PRECISION == :A7
    setprecision(32768)
    const T = BigFloat
    const PREV_PRECISION = :A6
    const NEXT_PRECISION = :A8
elseif PRECISION == :A8
    setprecision(65536)
    const T = BigFloat
    const PREV_PRECISION = :A7
    const NEXT_PRECISION = :A9
elseif PRECISION == :A9
    setprecision(131072)
    const T = BigFloat
    const PREV_PRECISION = :A8
    const NEXT_PRECISION = nothing
end


@static if PARAMETERIZATION == :AE
    const RKOCEvaluator = RKOCEvaluatorAE{T}
    const RKOCEvaluatorAdjoint = RungeKuttaToolKit.RKOCEvaluatorAEAdjoint{T}
    @inline num_parameters(s::Int) = (s * (s - 1)) >> 1
elseif PARAMETERIZATION == :AD
    const RKOCEvaluator = RKOCEvaluatorAD{T}
    const RKOCEvaluatorAdjoint = RungeKuttaToolKit.RKOCEvaluatorADAdjoint{T}
    @inline num_parameters(s::Int) = (s * (s + 1)) >> 1
elseif PARAMETERIZATION == :AI
    const RKOCEvaluator = RKOCEvaluatorAI{T}
    const RKOCEvaluatorAdjoint = RungeKuttaToolKit.RKOCEvaluatorAIAdjoint{T}
    @inline num_parameters(s::Int) = s * s
elseif PARAMETERIZATION == :BE
    const RKOCEvaluator = RKOCEvaluatorBE{T}
    const RKOCEvaluatorAdjoint = RungeKuttaToolKit.RKOCEvaluatorBEAdjoint{T}
    @inline num_parameters(s::Int) = (s * (s + 1)) >> 1
elseif PARAMETERIZATION == :BD
    const RKOCEvaluator = RKOCEvaluatorBD{T}
    const RKOCEvaluatorAdjoint = RungeKuttaToolKit.RKOCEvaluatorBDAdjoint{T}
    @inline num_parameters(s::Int) = (s * (s + 3)) >> 1
elseif PARAMETERIZATION == :BI
    const RKOCEvaluator = RKOCEvaluatorBI{T}
    const RKOCEvaluatorAdjoint = RungeKuttaToolKit.RKOCEvaluatorBIAdjoint{T}
    @inline num_parameters(s::Int) = s * (s + 1)
end


const RKOCOptimizer = LBFGSOptimizer{typeof(DZOptimization.NULL_CONSTRAINT),
    RKOCEvaluator,RKOCEvaluatorAdjoint,QuadraticLineSearch,T,1}


function compute_scores(optimizer::RKOCOptimizer)
    num_residuals = length(optimizer.objective_function.residuals)
    num_variables = length(optimizer.current_point)
    rms_residual = sqrt(optimizer.current_objective_value[] / num_residuals)
    rms_gradient = sqrt(norm2(optimizer.current_gradient) / num_variables)
    rms_coeff = sqrt(norm2(optimizer.current_point) / num_variables)
    residual_score = round(Int,
        clamp(-500 * log10(Float64(rms_residual)), 0.0, 9999.0))
    gradient_score = round(Int,
        clamp(-500 * log10(Float64(rms_gradient)), 0.0, 9999.0))
    coeff_score = round(Int,
        clamp(10000 - 2500 * log10(Float64(rms_coeff)), 0.0, 9999.0))
    return (residual_score, gradient_score, coeff_score)
end


function parse_number(str::AbstractString)
    try
        @static if T <: MultiFloat
            return T(str)
        else
            return parse(T, str)
        end
    catch
        println(stderr, "ERROR: Cannot parse $str as a floating-point number.")
        exit(EXIT_INVALID_NUMERIC_LITERAL)
    end
end


function read_blocks(path::AbstractString)
    if !isfile(path)
        println(stderr, "ERROR: Input file $path does not exist.")
        exit(EXIT_FILE_DOES_NOT_EXIST)
    end
    result = Vector{String}[]
    block = String[]
    for line in eachline(path)
        if all(isspace(c) for c in line)
            if !isempty(block)
                push!(result, block)
                block = String[]
            end
        else
            push!(block, line)
        end
    end
    if !isempty(block)
        push!(result, block)
    end
    return result
end


function parse_last_block(path::AbstractString)
    blocks = read_blocks(path)
    if isempty(blocks)
        println(stderr, "ERROR: Input file $path is empty.")
        exit(EXIT_INPUT_FILE_EMPTY)
    end
    return parse_number.(blocks[end])
end


function parse_last_block(path::AbstractString, stages::Int)
    result = parse_last_block(path)
    expected = num_parameters(stages)
    actual = length(result)
    if length(result) != num_parameters(stages)
        println(stderr,
            "ERROR: Input file $path contains an invalid number of entries" *
            " for parameterization $PARAMETERIZATION with $stages stages" *
            " (expected $expected, received $actual).")
        exit(EXIT_INVALID_PARAMETER_COUNT)
    end
    return result
end


function precision_is_sufficient(x::U, n::Int) where {U}
    s = @sprintf("%+.*e", n, x)
    return parse(U, s) == x
end


function precision_is_sufficient(x::MultiFloat{U,N}, n::Int) where {U,N}
    s = @sprintf("%+.*e", n, x)
    return MultiFloat{U,N}(s) == MultiFloats.renormalize(x)
end


precision_is_sufficient(x::Vector{U}, n::Int) where {U} =
    all(precision_is_sufficient(c, n) for c in x)


function find_sufficient_precision(x::Vector{U}) where {U}
    n = ceil(Int, precision(U; base=2) * log10(2.0))
    while true
        if precision_is_sufficient(x, n)
            return n
        end
        n += 1
    end
end


function uniform_precision_strings(x::Vector{U}) where {U}
    n = find_sufficient_precision(x)
    return [@sprintf("%+.*e", n, c) for c in x]
end
