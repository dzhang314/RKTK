# WARNING: This file should not be executed directly. It is intended to be
# included in other scripts using the include() function.


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
        "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"])
        println(stderr, "ERROR: Invalid mode string.")
        print(stderr, USAGE_STRING)
        exit(EXIT_INVALID_PRECISION)
    end
    return (Symbol(mode[1:2]), Symbol(mode[3:4]))
end


const PARAMETERIZATION, PRECISION = parse_mode(ARGS[1])


@static if PRECISION == :M1
    const T = Float64
elseif PRECISION == :M2
    const T = Float64x2
elseif PRECISION == :M3
    const T = Float64x3
elseif PRECISION == :M4
    const T = Float64x4
elseif PRECISION == :M5
    const T = Float64x5
elseif PRECISION == :M6
    const T = Float64x6
elseif PRECISION == :M7
    const T = Float64x7
elseif PRECISION == :M8
    const T = Float64x8
elseif PRECISION == :X1
    setprecision(512)
    const T = BigFloat
elseif PRECISION == :X2
    setprecision(1024)
    const T = BigFloat
elseif PRECISION == :X3
    setprecision(2048)
    const T = BigFloat
elseif PRECISION == :X4
    setprecision(4096)
    const T = BigFloat
elseif PRECISION == :X5
    setprecision(8192)
    const T = BigFloat
elseif PRECISION == :X6
    setprecision(16384)
    const T = BigFloat
elseif PRECISION == :X7
    setprecision(32768)
    const T = BigFloat
elseif PRECISION == :X8
    setprecision(65536)
    const T = BigFloat
elseif PRECISION == :X9
    setprecision(131072)
    const T = BigFloat
end


@static if PARAMETERIZATION == :AE
    const RKOCEvaluator = RKOCEvaluatorAE{T}
    @inline num_parameters(s::Int) = (s * (s - 1)) >> 1
elseif PARAMETERIZATION == :AD
    const RKOCEvaluator = RKOCEvaluatorAD{T}
    @inline num_parameters(s::Int) = (s * (s + 1)) >> 1
elseif PARAMETERIZATION == :AI
    const RKOCEvaluator = RKOCEvaluatorAI{T}
    @inline num_parameters(s::Int) = s * s
elseif PARAMETERIZATION == :BE
    const RKOCEvaluator = RKOCEvaluatorBE{T}
    @inline num_parameters(s::Int) = (s * (s + 1)) >> 1
elseif PARAMETERIZATION == :BD
    const RKOCEvaluator = RKOCEvaluatorBD{T}
    @inline num_parameters(s::Int) = (s * (s + 3)) >> 1
elseif PARAMETERIZATION == :BI
    const RKOCEvaluator = RKOCEvaluatorBI{T}
    @inline num_parameters(s::Int) = s * (s + 1)
end
