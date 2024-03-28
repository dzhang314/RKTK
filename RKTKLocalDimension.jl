using DZOptimization.Kernels: axpy!, dot, norm2


const USAGE_STRING = """
Usage: julia [jl_options] $PROGRAM_FILE <mode> <order> <stages> <filename>
[jl_options] refers to Julia options, such as -O3 or --threads=8.

<mode> is a four-character RKTK mode string, as explained in the RKTK manual.
<order> and <stages> must be positive integers between 1 and 99.
"""


include("./src/ParseMode.jl")


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

    trees = all_rooted_trees(ORDER; tree_ordering=TREE_ORDERING)
    evaluator = RKOCResidualEvaluator(trees, STAGES)

    x = parse_last_block(ARGS[4], STAGES)
    residuals = evaluator(x)
    _eps = eps(T)
    _sqrt_eps = sqrt(_eps)
    _loose_eps = _sqrt_eps
    _strict_eps = _sqrt_eps * sqrt(_sqrt_eps)
    if any(!(r * r < _eps) for r in residuals)
        println(stderr,
            "WARNING: Some residuals of this method significantly exceed" *
            " floating-point epsilon. The order conditions do not hold at" *
            " the requested level of precision. Results may be spurious.")
    end

    jacobian = evaluator'(x)
    m, n = size(jacobian)
    constraints = Dict{LevelSequence,Vector{T}}()
    for (i, tree) in enumerate(trees)
        constraints[tree] = jacobian[i, :]
    end

    result = Pair{LevelSequence,T}[]
    _zero = zero(T)
    while (length(result) < n) && !isempty(constraints)
        norm_squared, tree = findmax(norm2(v) for (t, v) in constraints)
        if !(norm_squared >= _zero)
            break
        end
        v = constraints[tree]
        push!(result, tree => sqrt(norm_squared))
        delete!(constraints, tree)
        for (_, w) in constraints
            axpy!(w, -dot(v, w, n) / norm_squared, v, n)
        end
    end

    tree_strings = [string(tree.data) * ':' for (tree, _) in result]
    tree_length = maximum(length(s) for s in tree_strings)
    norm_strings = uniform_lossy_strings(
        [norm for (_, norm) in result]; sign=false)
    num_zeros = 0
    num_uncertain = 0
    for (tree_string, norm_string, (_, norm)) in zip(
        tree_strings, norm_strings, result)
        if norm < _strict_eps
            println(rpad(tree_string, tree_length), ' ', norm_string, " (0)")
            num_zeros += 1
        elseif norm < _loose_eps
            println(rpad(tree_string, tree_length), ' ', norm_string, " (?)")
            num_uncertain += 1
        else
            println(rpad(tree_string, tree_length), ' ', norm_string)
        end
    end

    println("Jacobian dimensions: ($m order conditions, $n parameters)")
    if num_uncertain > 0
        println("Higher precision is needed to estimate the" *
                " rank of the Jacobian of the order conditions.")
    else
        estimated_rank = length(result) - num_zeros
        local_dimension = n - estimated_rank
        println("Estimated Jacobian rank: ", estimated_rank)
        println("Estimated local dimension: ", local_dimension)
    end

end


warn_single_threaded()
main()
