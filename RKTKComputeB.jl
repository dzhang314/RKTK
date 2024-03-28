const USAGE_STRING = """
Usage: julia [jl_options] $PROGRAM_FILE <mode> <order> <stages> <filename>
[jl_options] refers to Julia options, such as -O3 or --threads=8.

<mode> is a four-character RKTK mode string, as explained in the RKTK manual.
<order> and <stages> must be positive integers between 1 and 99.
"""


include("./src/ParseMode.jl")
if !startswith(string(PARAMETERIZATION), 'A')
    println(
        "ERROR: RKTKComputeB.jl can only run" *
        " using parameterization mode A.")
    exit(EXIT_INVALID_PARAMETERIZATION)
end


if length(ARGS) != 4
    print(stderr, USAGE_STRING)
    exit(EXIT_INVALID_ARG_COUNT)
end


const ORDER = tryparse(Int, ARGS[2]; base=10)
if isnothing(ORDER) || !(0 < ORDER < 100)
    println(stderr,
        "ERROR: $(ARGS[2]) is not a positive integer between 1 and 99.")
    print(stderr, USAGE_STRING)
    exit(EXIT_INVALID_ARG_FORMAT)
end


const STAGES = tryparse(Int, ARGS[3]; base=10)
if isnothing(STAGES) || !(0 < STAGES < 100)
    println(stderr,
        "ERROR: $(ARGS[3]) is not a positive integer between 1 and 99.")
    print(stderr, USAGE_STRING)
    exit(EXIT_INVALID_ARG_FORMAT)
end


function main()
    evaluator = RKOCEvaluator(ORDER, STAGES)
    x = parse_last_block(ARGS[4], STAGES)
    evaluator'(x)
    result = vcat(x, evaluator.b)
    for line in uniform_precision_strings(result)
        println(line)
    end
    # TODO: Write results to a file.
end


warn_single_threaded()
main()
