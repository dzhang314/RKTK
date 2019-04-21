using LinearAlgebra: I, qr

push!(LOAD_PATH, @__DIR__)
using MultiprecisionFloats
using RKTK2
using DZMisc

setprecision(5000)
const AccurateReal = BigFloat

function log(args...)
    println(args...)
    flush(stdout)
end

function constraint_residual(x)
    [x[1]^2 + x[2]^2 + x[3]^2 - one(AccurateReal), x[1] + x[2]^2]
end

function constraint_jacobian(x)
    [dbl(x[1]) dbl(x[2]) dbl(x[3]);
     one(AccurateReal) dbl(x[2]) zero(AccurateReal)]
end

function constraint_objective(x)
    norm2(constraint_residual(x))
end

function objective(x)
    x[1] * x[3]
end

function gradient(x)
    [x[3], zero(AccurateReal), x[1]]
end

function constrained_gradient(x)
    constraint_jac = copy(transpose(constraint_jacobian(x)))
    orthonormalize_columns!(constraint_jac)
    grad = gradient(x)
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
        objective(x_new)
    else
        AccurateReal(NaN)
    end
end

struct ConstrainedBFGSOptimizer
    objective_value::Ref{AccurateReal}
    last_step_size::Ref{AccurateReal}
end

const INPUT_POINT = AccurateReal[-0.5, 0.707107, -0.5]
const NUM_VARS = 3

const opt = ConstrainedBFGSOptimizer(
    objective(INPUT_POINT),
    AccurateReal(0.00001))

function main()

    x = copy(INPUT_POINT)
    inv_hess = Matrix{AccurateReal}(I, NUM_VARS, NUM_VARS)

    cons_grad = constrained_gradient(x)
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
            cons_grad_new = constrained_gradient(x)
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
            cons_grad = constrained_gradient(x)
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
