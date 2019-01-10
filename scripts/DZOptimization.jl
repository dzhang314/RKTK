module DZOptimization

export BFGSOptimizer, step!

using LinearAlgebra: mul!
using DZMisc: norm, quadratic_line_search, update_inverse_hessian!

struct BFGSOptimizer{T <: Real}
    n::Int
    x::Vector{T}
    temp::Vector{T}
    objective::Ref{T}
    gradient::Vector{T}
    last_step_size::Ref{T}
    delta_gradient::Vector{T}
    bfgs_dir::Vector{T}
    hess_inv::Matrix{T}
end

function BFGSOptimizer(vec::Vector{T}, step_size, f, g!) where {T <: Real}
    n = length(vec)
    x = Vector{T}(undef, n)
    @simd ivdep for i = 1 : n
        @inbounds x[i] = vec[i]
    end
    temp = Vector{T}(undef, n)
    objective = Ref{T}(f(x))
    gradient = Vector{T}(undef, n)
    g!(gradient, x)
    last_step_size = Ref{T}(step_size)
    delta_gradient = Vector{T}(undef, n)
    bfgs_dir = copy(gradient)
    hess_inv = Matrix{T}(undef, n, n)
    for j = 1 : n
        @simd ivdep for i = 1 : n
            @inbounds hess_inv[i, j] = T(i == j)
        end
    end
    BFGSOptimizer{T}(n, x, temp, objective, gradient,
        last_step_size, delta_gradient, bfgs_dir, hess_inv)
end

function reset_hessian!(opt::BFGSOptimizer{T}) where {T <: Real}
    for j = 1 : opt.n
        @simd ivdep for i = 1 : opt.n
            @inbounds opt.hess_inv[i, j] = T(i == j)
        end
    end
end

function step_obj(step_size::T, step_dir::Vector{T},
        opt::BFGSOptimizer{T}, f) where {T <: Real}
    @simd ivdep for i = 1 : opt.n
        @inbounds opt.temp[i] = opt.x[i] - step_size * step_dir[i]
    end
    f(opt.temp)
end

function step!(opt::BFGSOptimizer{T}, f, g!) where {T <: Real}
    step_size = opt.last_step_size[]
    objective = opt.objective[]
    grad_norm = norm(opt.gradient)
    bfgs_norm = norm(opt.bfgs_dir)
    grad_step_size, grad_obj = quadratic_line_search(step_obj,
        objective, step_size / grad_norm, opt.gradient, opt, f)
    bfgs_step_size, bfgs_obj = quadratic_line_search(step_obj,
        objective, step_size / bfgs_norm, opt.bfgs_dir, opt, f)
    if bfgs_obj <= grad_obj
        opt.objective[] = bfgs_obj
        objective_decreased = (bfgs_obj < objective)
        opt.last_step_size[] = bfgs_step_size * bfgs_norm
        @simd ivdep for i = 1 : opt.n
            @inbounds opt.x[i] -= bfgs_step_size * opt.bfgs_dir[i]
        end
        @simd ivdep for i = 1 : opt.n
            @inbounds opt.delta_gradient[i] = -opt.gradient[i]
        end
        g!(opt.gradient, opt.x)
        @simd ivdep for i = 1 : opt.n
            @inbounds opt.delta_gradient[i] += opt.gradient[i]
        end
        update_inverse_hessian!(opt.hess_inv, -bfgs_step_size, opt.bfgs_dir,
            opt.delta_gradient, opt.temp)
        mul!(opt.bfgs_dir, opt.hess_inv, opt.gradient)
        return true, objective_decreased
    else
        opt.objective[] = grad_obj
        objective_decreased = (grad_obj < objective)
        opt.last_step_size[] = grad_step_size * grad_norm
        @simd ivdep for i = 1 : opt.n
            @inbounds opt.x[i] -= grad_step_size * opt.gradient[i]
        end
        reset_hessian!(opt)
        g!(opt.gradient, opt.x)
        @simd ivdep for i = 1 : opt.n
            @inbounds opt.bfgs_dir[i] = opt.gradient[i]
        end
        return false, objective_decreased
    end
end

end # module DZOptimization
