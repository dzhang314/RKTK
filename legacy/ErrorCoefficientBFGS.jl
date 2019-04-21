using LinearAlgebra: I, qr
using Printf: @sprintf

push!(LOAD_PATH, @__DIR__)
using MultiprecisionFloats
using RKTK2
using DZMisc

const AccurateReal = Float64x3
const BASE_EVALUATOR = RKOCEvaluator{AccurateReal}(10, 16)
const ERROR_EVALUATOR = RKOCEvaluator{AccurateReal}(11, 16)

const NUM_VARS = BASE_EVALUATOR.num_vars
const NUM_CONSTRS = BASE_EVALUATOR.num_constrs
const NUM_ERR_COEFFS = ERROR_EVALUATOR.num_constrs

const CONSTR_IDXS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 31, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52,
    58, 59, 62, 75, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 106, 107, 110, 115, 116, 123, 134, 135, 201, 202, 203, 205, 206,
    207, 208, 210, 211, 214, 215, 221, 222, 225, 226, 227, 230, 231, 238, 249,
    250, 269, 270, 316, 404, 406, 487, 488, 489, 490, 491, 492, 493, 496, 497,
    498, 499, 500, 501, 502, 507, 508, 511, 516, 517, 524, 535, 536, 539, 544,
    555, 556, 602, 603, 650, 690, 984]

################################################################################

function log(args...)
    println(args...)
    flush(stdout)
end

function constraint_residual(x)
    result = Vector{AccurateReal}(undef, NUM_CONSTRS)
    evaluate_residual!(result, x, BASE_EVALUATOR)
    result[CONSTR_IDXS]
end

function constraint_jacobian(x)
    result = Matrix{AccurateReal}(undef, NUM_CONSTRS, NUM_VARS)
    evaluate_jacobian!(result, x, BASE_EVALUATOR)
    result[CONSTR_IDXS, :]
end

function constraint_objective(x)
    norm2(constraint_residual(x))
end

function error_coefficients(x)
    result = Vector{AccurateReal}(undef, NUM_ERR_COEFFS)
    evaluate_error_coefficients!(result, x, ERROR_EVALUATOR)
    result[NUM_CONSTRS+1:end]
end

function error_jacobian(x)
    result = Matrix{AccurateReal}(undef, NUM_ERR_COEFFS, NUM_VARS)
    evaluate_error_jacobian!(result, x, ERROR_EVALUATOR)
    result[NUM_CONSTRS+1:end, :]
end

function error_objective(x)
    norm2(error_coefficients(x))
end

function error_gradient(x)
    dbl.(transpose(error_jacobian(x)) * error_coefficients(x))
end

function constrained_error_gradient(x)
    constraint_jac = copy(transpose(constraint_jacobian(x)))
    orthonormalize_columns!(constraint_jac)
    grad = error_gradient(x)
    grad - constraint_jac * (transpose(constraint_jac) * grad)
end

function constrain_step(x, step)
    constraint_jac = copy(transpose(constraint_jacobian(x)))
    orthonormalize_columns!(constraint_jac)
    step - constraint_jac * (transpose(constraint_jac) * step)
end

################################################################################

function constrain(x)
    x_old, obj_old = x, constraint_objective(x)
    while true
        direction = qr(constraint_jacobian(x_old)) \ constraint_residual(x_old)
        x_new = x_old - direction
        obj_new = constraint_objective(x_new)
        if obj_new < obj_old
            x_old, obj_old = x_new, obj_new
        else
            break
        end
    end
    x_old, obj_old
end

function constrained_step_value(step_size, x, step_direction, step_norm)
    x_new, obj_new = constrain(x - (step_size / step_norm) * step_direction)
    if obj_new < AccurateReal(1e-75)
        error_objective(x_new)
    else
        AccurateReal(NaN)
    end
end

struct ConstrainedBFGSOptimizer
    objective_value::Ref{AccurateReal}
    last_step_size::Ref{AccurateReal}
end

const INPUT_FILE_NAME = readdir()[end]
log("Opening ", INPUT_FILE_NAME)
const INPUT_POINT = AccurateReal.(BigFloat.(split(read(INPUT_FILE_NAME, String))))
@assert(length(INPUT_POINT) == NUM_VARS)
log("Successfully read input file.")

const opt = ConstrainedBFGSOptimizer(
    error_objective(INPUT_POINT),
    AccurateReal(0.00001))

function main()

    x = copy(INPUT_POINT)
    inv_hess = Matrix{AccurateReal}(I, NUM_VARS, NUM_VARS)

    cons_grad = constrained_error_gradient(x)
    cons_grad_norm = norm(cons_grad)
    while true

        # log("Performing gradient descent step...")
        grad_step_size, obj_grad = quadratic_line_search(constrained_step_value,
            opt.objective_value[], opt.last_step_size[], x, cons_grad, cons_grad_norm)

        # log("Performing BFGS step...")
        bfgs_step = constrain_step(x, inv_hess * cons_grad)
        bfgs_step_norm = norm(bfgs_step)
        bfgs_step_size, obj_bfgs = quadratic_line_search(constrained_step_value,
            opt.objective_value[], opt.last_step_size[], x, bfgs_step, bfgs_step_norm)

        if obj_bfgs < opt.objective_value[] && obj_bfgs <= obj_grad
            x, _ = constrain(x - (bfgs_step_size / bfgs_step_norm) * bfgs_step)
            cons_grad_new = constrained_error_gradient(x)
            cons_grad_norm_new = norm(cons_grad_new)
            update_inverse_hessian!(inv_hess,
                -bfgs_step_size / bfgs_step_norm,
                bfgs_step,
                cons_grad_new - cons_grad,
                Vector{AccurateReal}(undef, NUM_VARS))
                opt.last_step_size[] = bfgs_step_size
                opt.objective_value[] = obj_bfgs
            cons_grad = cons_grad_new
            cons_grad_norm = cons_grad_norm_new
            log(Float64(opt.objective_value[]), " | ",
                Float64(cons_grad_norm), " | ",
                Float64(opt.last_step_size[]), " | BFGS")
        elseif obj_grad < opt.objective_value[]
            x, _ = constrain(x - (grad_step_size / cons_grad_norm) * cons_grad)
            inv_hess = Matrix{AccurateReal}(I, NUM_VARS, NUM_VARS)
            opt.last_step_size[] = grad_step_size
            opt.objective_value[] = obj_grad
            cons_grad = constrained_error_gradient(x)
            cons_grad_norm = norm(cons_grad)
            log(Float64(opt.objective_value[]), " | ",
                Float64(cons_grad_norm), " | ",
                Float64(opt.last_step_size[]), " | GRAD")
        else
            log("Neither BFGS step (", Float64(obj_bfgs),
                ") or gradient step (", Float64(obj_grad),
                ") are better than previous point (", Float64(opt.objective_value[]), ")")
            break
        end
    end
end

main()
