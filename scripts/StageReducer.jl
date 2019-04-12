using LinearAlgebra: I, dot, qr
using Printf

push!(LOAD_PATH, @__DIR__)
using RKTK2
using DZMisc
using MultiprecisionFloats

const AccurateReal = Float64x4
const THRESHOLD = AccurateReal(1e-50)

################################################################################

function say(args...)
    print("\33[2K\r")
    print(args...)
    flush(stdout)
end

function log(args...)
    print("\33[2K\r")
    println(args...)
    flush(stdout)
end

function numstr(x)
    @sprintf("%#-27.20g", BigFloat(x))
end

function shortstr(x)
    @sprintf("%#.5g", BigFloat(x))
end

function linearly_independent_column_indices!(
        mat::Matrix{T}, threshold::T) where {T <: Real}
    indices = Int[]
    lo, hi, m, n = zero(T), T(Inf), size(mat, 1), size(mat, 2)
    for i = 1 : n
        vec = mat[:, i]
        x = norm(vec)
        if x > threshold
            hi = min(hi, x)
            normalize!(vec)
            push!(indices, i)
            for j = i + 1 : n
                acc = zero(T)
                for k = 1 : m
                    acc += vec[k] * mat[k, j]
                end
                for k = 1 : m
                    mat[k, j] -= acc * vec[k]
                end
            end
        else
            lo = max(lo, x)
        end
    end
    indices, lo, hi
end

function constrain(x::Vector{T}, evaluator::RKOCEvaluator{T}) where {T <: Real}
    x_old, obj_old = x, norm2(evaluator(x))
    while true
        direction = qr(evaluator'(x_old)) \ evaluator(x_old)
        x_new = x_old - direction
        obj_new = norm2(evaluator(x_new))
        if obj_new < obj_old
            x_old, obj_old = x_new, obj_new
        else
            break
        end
    end
    x_old, obj_old
end

function compute_stages(x::Vector{T}) where {T <: Real}
    num_stages = div(isqrt(8 * length(x) + 1) - 1, 2)
    @assert(length(x) == div(num_stages * (num_stages + 1), 2))
    num_stages
end

function compute_order(x::Vector{T}, threshold::T) where {T <: Real}
    num_stages = compute_stages(x)
    order = 2
    while true
        say("    Testing constraints for order ", order, "...")
        x_new, obj_new = constrain(x,
            RKOCEvaluator{AccurateReal}(order, num_stages))
        if obj_new <= threshold^2
            x = x_new
            order += 1
        else
            break
        end
    end
    x, order - 1
end

function drop_last_stage(x::Vector{T}) where {T <: Real}
    num_stages = compute_stages(x)
    vcat(x[1 : div((num_stages - 1) * (num_stages - 2), 2)],
         x[div(num_stages * (num_stages - 1), 2) + 1 : end - 1])
end

################################################################################

# function objective(x)
#     x[end]^2
# end

# function gradient(x)
#     result = zero(x)
#     result[end] = dbl(x[end])
#     result
# end

function constrain_step(x, step, evaluator)
    constraint_jacobian = copy(transpose(evaluator'(x)))
    orthonormalize_columns!(constraint_jacobian)
    step - constraint_jacobian * (transpose(constraint_jacobian) * step)
end

function constrained_step_value(step_size,
        x, step_direction, step_norm, evaluator, threshold)
    x_new, obj_new = constrain(
        x - (step_size / step_norm) * step_direction, evaluator)
    if obj_new < threshold
        objective(x_new)
    else
        AccurateReal(NaN)
    end
end

################################################################################

struct ConstrainedBFGSOptimizer{T}
    objective_value::Ref{T}
    last_step_size::Ref{T}
end

################################################################################

log("                ______ _   _______ _   __")
log("                | ___ \\ | / /_   _| | / /   Version  2.1")
log("                | |_/ / |/ /  | | | |/ /")
log("                |    /|    \\  | | |    \\   David K. Zhang")
log("                | |\\ \\| |\\  \\ | | | |\\  \\     (c) 2019")
log("                |_| \\_\\_| \\_/ |_| |_| \\_/")
log()
log("RKTK is free software distributed under the terms of the MIT license.")
log()

if length(ARGS) != 1
    log("Usage: julia StageReducer.jl <input-file>")
    exit()
end

const INPUT_POINT = AccurateReal.(BigFloat.(split(read(ARGS[1], String))))
log("Successfully read input file: " * ARGS[1])

const NUM_VARS = length(INPUT_POINT)
const NUM_STAGES = compute_stages(INPUT_POINT)
const REFINED_POINT, ORDER = compute_order(INPUT_POINT, THRESHOLD)
log("    ", NUM_STAGES, "-stage method of order ", ORDER,
    " (refined by ", approx_norm(REFINED_POINT - INPUT_POINT), ").")

const FULL_CONSTRAINTS = RKOCEvaluator{AccurateReal}(ORDER, NUM_STAGES)
const ACTIVE_CONSTRAINT_INDICES, HI, LO = linearly_independent_column_indices!(
    copy(transpose(FULL_CONSTRAINTS'(INPUT_POINT))), THRESHOLD)
const ACTIVE_CONSTRAINTS = RKOCEvaluator{AccurateReal}(
    ACTIVE_CONSTRAINT_INDICES, NUM_STAGES)
log("    ", ACTIVE_CONSTRAINTS.num_constrs, " out of ",
    FULL_CONSTRAINTS.num_constrs, " active constraints.")
log("    Constraint thresholds: [",
    shortstr(-log2(BigFloat(LO))), " | ",
    shortstr(-log2(BigFloat(THRESHOLD))), " | ",
    shortstr(-log2(BigFloat(HI))), "]")
log()

const ERROR_EVALUATOR = RKOCEvaluator{AccurateReal}(
    Vector{Int}(rooted_tree_count(ORDER) + 1 : rooted_tree_count(ORDER + 1)),
    NUM_STAGES)

function objective(x)
    norm2(ERROR_EVALUATOR(x))
end

function gradient(x)
    dbl.(transpose(ERROR_EVALUATOR'(x)) * ERROR_EVALUATOR(x))
end

const OPT = ConstrainedBFGSOptimizer{AccurateReal}(
    objective(REFINED_POINT),
    AccurateReal(0.00001))

function print_table_header()
    log("       Objective value       │  Constrained gradient norm  │",
        "       Last step size        │ Type")
    log("─────────────────────────────┼─────────────────────────────┼",
        "─────────────────────────────┼──────")
end

function print_table_row(obj_value, grad_norm, step_size, type)
    log(" ",
        numstr(obj_value), " │ ",
        numstr(grad_norm), " │ ",
        numstr(step_size), " │ ",
        type)
end


function main()
    x = copy(REFINED_POINT)
    inv_hess = Matrix{AccurateReal}(I, NUM_VARS, NUM_VARS)

    cons_grad = constrain_step(x, gradient(x), ACTIVE_CONSTRAINTS)
    cons_grad_norm = norm(cons_grad)
    print_table_header()
    print_table_row(OPT.objective_value[], cons_grad_norm, 0, "NONE")

    while true

        say("Performing gradient descent step...")
        grad_step_size, obj_grad = quadratic_search(constrained_step_value,
            OPT.last_step_size[], x, cons_grad, cons_grad_norm,
            ACTIVE_CONSTRAINTS, THRESHOLD)

        say("Performing BFGS step...")
        bfgs_step = constrain_step(x, inv_hess * cons_grad, ACTIVE_CONSTRAINTS)
        bfgs_step_norm = norm(bfgs_step)
        bfgs_step_size, obj_bfgs = quadratic_search(constrained_step_value,
            OPT.last_step_size[], x, bfgs_step, bfgs_step_norm,
            ACTIVE_CONSTRAINTS, THRESHOLD)

        say("Line searches complete.")
        if obj_bfgs < OPT.objective_value[] && obj_bfgs <= obj_grad
            x, _ = constrain(x - (bfgs_step_size / bfgs_step_norm) * bfgs_step,
            ACTIVE_CONSTRAINTS)
            cons_grad_new = constrain_step(x, gradient(x), ACTIVE_CONSTRAINTS)
            cons_grad_norm_new = norm(cons_grad_new)
            update_inverse_hessian!(inv_hess,
                -bfgs_step_size / bfgs_step_norm,
                bfgs_step,
                cons_grad_new - cons_grad,
                Vector{AccurateReal}(undef, NUM_VARS))
                OPT.last_step_size[] = bfgs_step_size
                OPT.objective_value[] = obj_bfgs
            cons_grad = cons_grad_new
            cons_grad_norm = cons_grad_norm_new
            print_table_row(OPT.objective_value[], cons_grad_norm,
                OPT.last_step_size[], "")
        elseif obj_grad < OPT.objective_value[]
            x, _ = constrain(x - (grad_step_size / cons_grad_norm) * cons_grad,
                ACTIVE_CONSTRAINTS)
            inv_hess = Matrix{AccurateReal}(I, NUM_VARS, NUM_VARS)
            OPT.last_step_size[] = grad_step_size
            OPT.objective_value[] = obj_grad
            cons_grad = constrain_step(x, gradient(x), ACTIVE_CONSTRAINTS)
            cons_grad_norm = norm(cons_grad)
            print_table_row(OPT.objective_value[], cons_grad_norm,
                OPT.last_step_size[], "GRAD")
        else
            print_table_row(OPT.objective_value[], cons_grad_norm, 0, "DONE")
            log()
            break
        end

    end

    println.(string.(BigFloat.(x)))

    # x_new = drop_last_stage(x)
    # println.(string.(BigFloat.(x_new)))
end

main()
