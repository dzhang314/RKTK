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


function b_mask(i::Int)
    result = zeros(T, num_parameters(STAGES))
    result[b_index(STAGES, i)] = one(T)
    return result
end


function c_mask(i::Int)
    result = zeros(T, num_parameters(STAGES))
    _one = one(T)
    for j in c_range(STAGES, i)
        result[j] = _one
    end
    return result
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

    result = Pair{LevelSequence,Tuple{T,Vector{T}}}[]
    _zero = zero(T)
    while (length(result) < n) && !isempty(constraints)
        norm_squared, tree = findmax(norm2(v) for (t, v) in constraints)
        if !(norm_squared >= _zero)
            break
        end
        v = constraints[tree]
        push!(result, tree => (sqrt(norm_squared), v))
        delete!(constraints, tree)
        for (_, w) in constraints
            axpy!(w, -dot(v, w, n) / norm_squared, v, n)
        end
    end

    num_zeros = 0
    num_uncertain = 0
    for (_, (norm, _)) in result
        if norm < _strict_eps
            num_zeros += 1
        elseif norm < _loose_eps
            num_uncertain += 1
        end
    end

    if num_uncertain > 0
        println("Higher precision is needed to estimate the" *
                " rank of the Jacobian of the order conditions.")
        return nothing
    end

    estimated_rank = length(result) - num_zeros
    local_dimension = n - estimated_rank
    println("Estimated local dimension of method family: ", local_dimension)

    left_sides = String[]
    right_sides = T[]

    k = 1
    for i = 2:STAGES
        for j = 1:i-1
            v = zeros(T, num_parameters(STAGES))
            v[k] = one(T)
            for q = 1:estimated_rank
                (_, (_, w)) = result[q]
                axpy!(v, -dot(v, w, n) / norm2(w), w, n)
            end
            norm = sqrt(norm2(v))
            if norm < _strict_eps
                push!(left_sides, "    A[$i, $j]:")
                push!(right_sides, x[k])
            elseif norm < _loose_eps
                push!(left_sides, "(?) A[$i, $j]:")
                push!(right_sides, x[k])
            end
        end
    end

    for i = 1:STAGES
        v = b_mask(i)
        for j = 1:estimated_rank
            (_, (_, w)) = result[j]
            axpy!(v, -dot(v, w, n) / norm2(w), w, n)
        end
        norm = sqrt(norm2(v))
        if norm < _strict_eps
            push!(left_sides, "    b[$i]:")
            push!(right_sides, dot(b_mask(i), x, n))
        elseif norm < _loose_eps
            push!(left_sides, "(?) b[$i]:")
            push!(right_sides, dot(b_mask(i), x, n))
        end
    end

    for i = 1:STAGES-1
        for j = i+1:STAGES
            v = b_mask(i) - b_mask(j)
            for k = 1:estimated_rank
                (_, (_, w)) = result[k]
                axpy!(v, -dot(v, w, n) / norm2(w), w, n)
            end
            norm = sqrt(norm2(v))
            if norm < _strict_eps
                push!(left_sides, "    b[$i] - b[$j]:")
                push!(right_sides, dot(b_mask(i) - b_mask(j), x, n))
            elseif norm < _loose_eps
                push!(left_sides, "(?) b[$i] - b[$j]:")
                push!(right_sides, dot(b_mask(i) - b_mask(j), x, n))
            end
        end
    end

    for i = 1:STAGES
        v = c_mask(i)
        for j = 1:estimated_rank
            (_, (_, w)) = result[j]
            axpy!(v, -dot(v, w, n) / norm2(w), w, n)
        end
        norm = sqrt(norm2(v))
        if norm < _strict_eps
            push!(left_sides, "    c[$i]:")
            push!(right_sides, dot(c_mask(i), x, n))
        elseif norm < _loose_eps
            push!(left_sides, "(?) c[$i]:")
            push!(right_sides, dot(c_mask(i), x, n))
        end
    end

    for i = 1:STAGES-1
        for j = i+1:STAGES
            v = c_mask(i) - c_mask(j)
            for k = 1:estimated_rank
                (_, (_, w)) = result[k]
                axpy!(v, -dot(v, w, n) / norm2(w), w, n)
            end
            norm = sqrt(norm2(v))
            if norm < _strict_eps
                push!(left_sides, "    c[$i] - c[$j]:")
                push!(right_sides, dot(c_mask(i) - c_mask(j), x, n))
            elseif norm < _loose_eps
                push!(left_sides, "(?) c[$i] - c[$j]:")
                push!(right_sides, dot(c_mask(i) - c_mask(j), x, n))
            end
        end
    end

    left_length = maximum(length(s) for s in left_sides)
    right_strings = uniform_lossy_strings(right_sides)
    for (left, right) in zip(left_sides, right_strings)
        println(rpad(left, left_length), ' ', right)
    end

end


main()
