# WARNING: This file should not be executed directly. It should be included in
# other RKTK scripts using the include() function in order to parse the first
# command-line argument, which specifies the RKTK operating mode.


using Base.Threads
using DZOptimization
using DZOptimization.PCG
using MultiFloats
using Printf
using RungeKuttaToolKit


function warn_single_threaded()
    if nthreads() != 1
        println(stderr,
            "WARNING: Multiple threads of execution were requested, but" *
            " $PROGRAM_FILE is a single-threaded program. Execution will" *
            " continue using only one thread.")
    end
end


function get_tree_ordering()
    requested_orderings = [arg[17:end] for arg in ARGS if (
        startswith(arg, "--tree-ordering=") ||
        startswith(arg, "--tree_ordering="))]
    if length(requested_orderings) > 1
        println(stderr, "ERROR: Only one tree ordering can be specified.")
        exit(EXIT_INVALID_TREE_ORDERING)
    end
    if isempty(requested_orderings)
        return :reverse_lexicographic
    end
    result = Symbol(only(requested_orderings))
    if !(result in [:lexicographic, :reverse_lexicographic])
        println(stderr,
            "ERROR: Unknown tree ordering specified." *
            " Allowed values are lexicographic and reverse_lexicographic.")
        exit(EXIT_INVALID_TREE_ORDERING)
    end
    filter!(arg -> !(
            startswith(arg, "--tree-ordering=") ||
            startswith(arg, "--tree_ordering=")), ARGS)
    return result
end


const TREE_ORDERING = get_tree_ordering()


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
    const RKOCResidualEvaluator = RKOCResidualEvaluatorAE{T}
    const RKOCResidualEvaluatorAdjoint =
        RungeKuttaToolKit.RKOCResidualEvaluatorAEAdjoint{T}
    @inline num_parameters(s::Int) = (s * (s - 1)) >> 1
    @inline b_index(s::Int, i::Int) = ((s * (s - 1)) >> 1) + i
    @inline c_range(::Int, i::Int) = (((i*(i-3))>>1)+2):((i*(i-1))>>1)
elseif PARAMETERIZATION == :AD
    const RKOCEvaluator = RKOCEvaluatorAD{T}
    const RKOCEvaluatorAdjoint = RungeKuttaToolKit.RKOCEvaluatorADAdjoint{T}
    const RKOCResidualEvaluator = RKOCResidualEvaluatorAD{T}
    const RKOCResidualEvaluatorAdjoint =
        RungeKuttaToolKit.RKOCResidualEvaluatorADAdjoint{T}
    @inline num_parameters(s::Int) = (s * (s + 1)) >> 1
    @inline b_index(s::Int, i::Int) = ((s * (s + 1)) >> 1) + i
    @inline c_range(::Int, i::Int) = (((i*(i-1))>>1)+1):((i*(i+1))>>1)
elseif PARAMETERIZATION == :AI
    const RKOCEvaluator = RKOCEvaluatorAI{T}
    const RKOCEvaluatorAdjoint = RungeKuttaToolKit.RKOCEvaluatorAIAdjoint{T}
    const RKOCResidualEvaluator = RKOCResidualEvaluatorAI{T}
    const RKOCResidualEvaluatorAdjoint =
        RungeKuttaToolKit.RKOCResidualEvaluatorAIAdjoint{T}
    @inline num_parameters(s::Int) = s * s
    @inline b_index(s::Int, i::Int) = s * s + i
    @inline c_range(s::Int, i::Int) = (s*(i-1)+1):(s*i)
elseif PARAMETERIZATION == :BE
    const RKOCEvaluator = RKOCEvaluatorBE{T}
    const RKOCEvaluatorAdjoint = RungeKuttaToolKit.RKOCEvaluatorBEAdjoint{T}
    const RKOCResidualEvaluator = RKOCResidualEvaluatorBE{T}
    const RKOCResidualEvaluatorAdjoint =
        RungeKuttaToolKit.RKOCResidualEvaluatorBEAdjoint{T}
    @inline num_parameters(s::Int) = (s * (s + 1)) >> 1
    @inline b_index(s::Int, i::Int) = ((s * (s - 1)) >> 1) + i
    @inline c_range(::Int, i::Int) = (((i*(i-3))>>1)+2):((i*(i-1))>>1)
elseif PARAMETERIZATION == :BD
    const RKOCEvaluator = RKOCEvaluatorBD{T}
    const RKOCEvaluatorAdjoint = RungeKuttaToolKit.RKOCEvaluatorBDAdjoint{T}
    const RKOCResidualEvaluator = RKOCResidualEvaluatorBD{T}
    const RKOCResidualEvaluatorAdjoint =
        RungeKuttaToolKit.RKOCResidualEvaluatorBDAdjoint{T}
    @inline num_parameters(s::Int) = (s * (s + 3)) >> 1
    @inline b_index(s::Int, i::Int) = ((s * (s + 1)) >> 1) + i
    @inline c_range(::Int, i::Int) = (((i*(i-1))>>1)+1):((i*(i+1))>>1)
elseif PARAMETERIZATION == :BI
    const RKOCEvaluator = RKOCEvaluatorBI{T}
    const RKOCEvaluatorAdjoint = RungeKuttaToolKit.RKOCEvaluatorBIAdjoint{T}
    const RKOCResidualEvaluator = RKOCResidualEvaluatorBI{T}
    const RKOCResidualEvaluatorAdjoint =
        RungeKuttaToolKit.RKOCResidualEvaluatorBIAdjoint{T}
    @inline num_parameters(s::Int) = s * (s + 1)
    @inline b_index(s::Int, i::Int) = s * s + i
    @inline c_range(s::Int, i::Int) = (s*(i-1)+1):(s*i)
end


const RKOCOptimizer = LBFGSOptimizer{typeof(DZOptimization.NULL_CONSTRAINT),
    RKOCEvaluator,RKOCEvaluatorAdjoint,QuadraticLineSearch,T,1}


function create_optimizer(evaluator::RKOCEvaluator, stages::Int, seed::UInt64)
    n = num_parameters(stages)
    return LBFGSOptimizer(evaluator, evaluator', QuadraticLineSearch(),
        random_array(seed, T, n), sqrt(n * eps(T)), n)
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
