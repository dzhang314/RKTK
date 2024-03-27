using DZOptimization.Kernels: norm2


const USAGE_STRING = """
Usage: julia [jl_opts] $PROGRAM_FILE <mode> <order> <stages> <filename> [opts]
[jl_opts] refers to Julia options, such as -O3 or --threads=8.

<mode> is a four-character RKTK mode string, as explained in the RKTK manual.
<order> and <stages> must be positive integers between 1 and 99.

Available options:
    --all-orders
    --tree-ordering=lexicographic
    --tree-ordering=reverse_lexicographic
"""


include("./src/ParseMode.jl")


const ALL_ORDERS = ("--all-orders" in ARGS) || ("--all_orders" in ARGS)
filter!(arg -> !(arg == "--all-orders" || arg == "--all_orders"), ARGS)


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

    trees = ALL_ORDERS ?
            all_rooted_trees(ORDER; tree_ordering=TREE_ORDERING) :
            rooted_trees(ORDER, tree_ordering=TREE_ORDERING)
    evaluator = RKOCResidualEvaluator(trees, STAGES)

    x = parse_last_block(ARGS[4], STAGES)
    residuals = evaluator(x)
    errors = [residual / butcher_symmetry(tree)
              for (tree, residual) in zip(trees, residuals)]
    tree_strings = [string(tree.data) * ':' for tree in trees]
    tree_length = maximum(length(s) for s in tree_strings)
    error_strings = uniform_precision_strings(errors)
    for (tree_string, error_string) in zip(tree_strings, error_strings)
        println(rpad(tree_string, tree_length), ' ', error_string)
    end
    println("RMS Principal Error: ", sqrt(norm2(errors) / length(errors)))

end


main()
