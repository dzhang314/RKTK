module DZOptimization

export BFGSOptimizer, step!

using LinearAlgebra: mul!
using DZMisc: norm, dot, quadratic_line_search, identity_matrix!

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

function BFGSOptimizer(vec::Vector{T}, step_size,
        f, g!, args...) where {T <: Real}
    n = length(vec)
    x = Vector{T}(undef, n)
    @simd ivdep for i = 1 : n
        @inbounds x[i] = vec[i]
    end
    temp = Vector{T}(undef, n)
    objective = Ref{T}(f(x, args...))
    gradient = Vector{T}(undef, n)
    g!(gradient, x, args...)
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

function step_obj(step_size::T, step_dir::Vector{T},
        opt::BFGSOptimizer{T}, f, args...) where {T <: Real}
    @simd ivdep for i = 1 : opt.n
        @inbounds opt.temp[i] = opt.x[i] - step_size * step_dir[i]
    end
    f(opt.temp, args...)
end

function update_inverse_hessian!(B_inv::Matrix{T}, h::T,
        s::Vector{T}, y::Vector{T}, t::Vector{T})::Nothing where {T <: Real}
    b = dot(s, y)
    s .*= inv(b)
    mul!(t, B_inv, y)
    a = h * b + dot(y, t)
    for j = 1 : size(B_inv, 2)
        sj = s[j]
        tj = t[j]
        @simd ivdep for i = 1 : size(B_inv, 1)
            @inbounds B_inv[i, j] += a * (s[i] * sj) - (t[i] * sj + s[i] * tj)
        end
    end
end

function step!(opt::BFGSOptimizer{T}, f, g!, args...) where {T <: Real}
    step_size, objective = opt.last_step_size[], opt.objective[]
    grad_dir, bfgs_dir = opt.gradient, opt.bfgs_dir
    grad_norm, bfgs_norm = norm(grad_dir), norm(bfgs_dir)
    delta_grad, hess_inv = opt.delta_gradient, opt.hess_inv
    x, n = opt.x, opt.n
    grad_step_size, grad_obj = quadratic_line_search(step_obj,
        objective, step_size / grad_norm, grad_dir, opt, f, args...)
    bfgs_step_size, bfgs_obj = quadratic_line_search(step_obj,
        objective, step_size / bfgs_norm, bfgs_dir, opt, f, args...)
    if bfgs_obj <= grad_obj
        opt.objective[] = bfgs_obj
        objective_decreased = (bfgs_obj < objective)
        opt.last_step_size[] = bfgs_step_size * bfgs_norm
        @simd ivdep for i = 1 : n
            @inbounds x[i] -= bfgs_step_size * bfgs_dir[i]
        end
        @simd ivdep for i = 1 : n
            @inbounds delta_grad[i] = -grad_dir[i]
        end
        g!(grad_dir, x, args...)
        @simd ivdep for i = 1 : n
            @inbounds delta_grad[i] += grad_dir[i]
        end
        update_inverse_hessian!(hess_inv, -bfgs_step_size, bfgs_dir,
            delta_grad, opt.temp)
        mul!(bfgs_dir, hess_inv, grad_dir)
        return true, objective_decreased
    else
        opt.objective[] = grad_obj
        objective_decreased = (grad_obj < objective)
        opt.last_step_size[] = grad_step_size * grad_norm
        @simd ivdep for i = 1 : n
            @inbounds x[i] -= grad_step_size * grad_dir[i]
        end
        identity_matrix!(hess_inv)
        g!(grad_dir, x, args...)
        @simd ivdep for i = 1 : n
            @inbounds bfgs_dir[i] = grad_dir[i]
        end
        return false, objective_decreased
    end
end

end # module DZOptimization
