module DZOptimization

export BFGSOptimizer, step!

using LinearAlgebra: mul!
using DZMisc: scale, norm, dot, identity_matrix!

################################################################################

@inline function _qls_best(fb::T, x1::T, f1::T, x2::T, f2::T,
                           x3::T, f3::T)::Tuple{T,T} where {T <: Real}
    xb = zero(T)
    if f1 < fb; xb, fb = x1, f1; end
    if f2 < fb; xb, fb = x2, f2; end
    if f3 < fb; xb, fb = x3, f3; end
    xb, fb
end

@inline function _qls_minimum_high(f0::T, f1::T, f2::T)::T where {T <: Number}
    q1 = f1 + f1
    q2 = q1 + q1
    q3 = f0 + f0
    q4 = f2 + f0
    q5 = q1 - q4
    (q2 - q3 - q4) / (q5 + q5)
end

@inline function _qls_minimum_low(f0::T, f1::T, f2::T)::T where {T <: Number}
    q1 = f2 + f2
    q2 = q1 + q1
    q3 = f0 + f0
    q4 = f0 + f1
    q5 = q4 - q1
    q6 = q5 + q5
    (q4 + q3 - q2) / (q6 + q6)
end

function quadratic_line_search(f::S, f0::T,
                               x1::T)::Tuple{T,T} where {S, T <: Real}
    if isnan(f0)
        return zero(T), f0
    end
    f1 = f(x1)
    while isnan(f1)
        x1 = scale(0.5, x1)
        f1 = f(x1)
    end
    if f1 < f0
        while true
            x2 = scale(2.0, x1)
            f2 = f(x2)
            if (f2 >= f1) || isnan(f2)
                x3 = x1 * _qls_minimum_high(f0, f1, f2)
                f3 = f(x3)
                return _qls_best(f0, x1, f1, x2, f2, x3, f3)
            else
                x1, f1 = x2, f2
            end
        end
    else
        while true
            x2 = scale(0.5, x1)
            f2 = f(x2)
            if isnan(f2)
                return zero(T), f0
            end
            if f2 <= f0
                x3 = x1 * _qls_minimum_low(f0, f1, f2)
                f3 = f(x3)
                return _qls_best(f0, x2, f2, x1, f1, x3, f3)
            else
                x1, f1 = x2, f2
            end
        end
    end
end

function update_inverse_hessian!(B_inv::Matrix{T}, h::T, s::Vector{T},
        y::Vector{T}, t::Vector{T})::Nothing where {T <: Real}
    b = dot(s, y)
    s .*= inv(b)
    mul!(t, B_inv, y)
    a = h * b + dot(y, t)
    n = size(B_inv, 1)
    @inbounds for j = 1 : n
        sj = s[j]
        tj = t[j]
        @simd ivdep for i = 1 : n
            B_inv[i, j] += a * (s[i] * sj) - (t[i] * sj + s[i] * tj)
        end
    end
end

################################################################################

struct StepObjectiveFunctor{S, T <: Real}
    objective_functor::S
    base_point::Vector{T}
    step_point::Vector{T}
    step_direction::Vector{T}
end

function (so::StepObjectiveFunctor{S,T})(step_size::T)::T where {S, T <: Real}
    x, step, dx = so.base_point, so.step_point, so.step_direction
    n = length(x)
    @simd ivdep for i = 1 : n
        @inbounds step[i] = x[i] - step_size * dx[i]
    end
    so.objective_functor(step)
end

################################################################################

struct BFGSOptimizer{S1, S2, T <: Real}
    num_dims::Int
    objective_functor::S1
    gradient_functor!::S2
    current_point::Vector{T}
    temp_buffer::Vector{T}
    objective::Vector{T}
    gradient::Vector{T}
    last_step_size::Vector{T}
    delta_gradient::Vector{T}
    bfgs_dir::Vector{T}
    hess_inv::Matrix{T}
    grad_functor::StepObjectiveFunctor{S1,T}
    bfgs_functor::StepObjectiveFunctor{S1,T}
end

function BFGSOptimizer(initial_point::Vector{T}, initial_step_size::T,
        objective_functor::S1, gradient_functor!::S2) where {S1, S2, T <: Real}
    num_dims = length(initial_point)
    current_point = copy(initial_point)
    temp_buffer = Vector{T}(undef, num_dims)
    objective = objective_functor(current_point)
    gradient = Vector{T}(undef, num_dims)
    gradient_functor!(gradient, current_point)
    bfgs_dir = copy(gradient)
    hess_inv = Matrix{T}(undef, num_dims, num_dims)
    identity_matrix!(hess_inv)
    BFGSOptimizer{S1,S2,T}(num_dims, objective_functor, gradient_functor!,
        current_point, temp_buffer, T[objective], gradient,
        T[initial_step_size], Vector{T}(undef, num_dims),
        bfgs_dir, hess_inv,
        StepObjectiveFunctor{S1,T}(objective_functor, current_point,
            temp_buffer, gradient),
        StepObjectiveFunctor{S1,T}(objective_functor, current_point,
            temp_buffer, bfgs_dir))
end

function step!(opt::BFGSOptimizer{S1,S2,T}) where {S1, S2, T <: Real}
    @inbounds step_size, objective = opt.last_step_size[1], opt.objective[1]
    grad_dir, bfgs_dir = opt.gradient, opt.bfgs_dir
    delta_grad, hess_inv = opt.delta_gradient, opt.hess_inv
    x, n = opt.current_point, opt.num_dims
    grad_norm, bfgs_norm = norm(grad_dir), norm(bfgs_dir)
    grad_step_size, grad_obj = quadratic_line_search(
        opt.grad_functor, objective, step_size / grad_norm)
    bfgs_step_size, bfgs_obj = quadratic_line_search(
        opt.bfgs_functor, objective, step_size / bfgs_norm)
    if bfgs_obj <= grad_obj
        @inbounds opt.objective[1] = bfgs_obj
        objective_decreased = (bfgs_obj < objective)
        @inbounds opt.last_step_size[1] = bfgs_step_size * bfgs_norm
        @simd ivdep for i = 1 : n
            @inbounds x[i] -= bfgs_step_size * bfgs_dir[i]
        end
        @simd ivdep for i = 1 : n
            @inbounds delta_grad[i] = -grad_dir[i]
        end
        opt.gradient_functor!(grad_dir, x)
        @simd ivdep for i = 1 : n
            @inbounds delta_grad[i] += grad_dir[i]
        end
        update_inverse_hessian!(hess_inv, -bfgs_step_size, bfgs_dir,
            delta_grad, opt.temp_buffer)
        mul!(bfgs_dir, hess_inv, grad_dir)
        true, objective_decreased
    else
        @inbounds opt.objective[1] = grad_obj
        objective_decreased = (grad_obj < objective)
        @inbounds opt.last_step_size[1] = grad_step_size * grad_norm
        @simd ivdep for i = 1 : n
            @inbounds x[i] -= grad_step_size * grad_dir[i]
        end
        identity_matrix!(hess_inv)
        opt.gradient_functor!(grad_dir, x)
        @simd ivdep for i = 1 : n
            @inbounds bfgs_dir[i] = grad_dir[i]
        end
        false, objective_decreased
    end
end

end # module DZOptimization
