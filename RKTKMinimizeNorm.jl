using LinearAlgebra: mul!
using Printf: @printf

using MultiFloats
using DZOptimization
using DZOptimization.Kernels: norm2
using RungeKuttaToolKit
using RungeKuttaToolKit.ButcherInstructions: LevelSequence


function rank_revealing_gram_schmidt!(
    Q::AbstractMatrix{T}, threshold::T
) where {T}
    t, s = size(Q)
    active = Vector{Pair{Int,T}}()
    inactive = Vector{Pair{Int,T}}()
    _zero = zero(T)
    @inbounds for i = 1:s
        squared_norm = _zero
        for k = 1:t
            squared_norm += abs2(Q[k, i])
        end
        if squared_norm < threshold
            push!(inactive, i => squared_norm)
        else
            push!(active, i => squared_norm)
            inv_norm = RungeKuttaToolKit.inv_sqrt(squared_norm)
            @simd ivdep for k = 1:t
                Q[k, i] *= inv_norm
            end
            for j = i+1:s
                overlap = _zero
                for k = 1:t
                    overlap += Q[k, i] * Q[k, j]
                end
                @simd ivdep for k = 1:t
                    Q[k, j] -= overlap * Q[k, i]
                end
            end
        end
    end
    return (active, inactive)
end


function constrain!(
    x::AbstractVector{T}, temp::AbstractVector{T},
    ev::RKOCResidualEvaluatorBE{T},
    residual::AbstractVector{T}, jacobian::AbstractMatrix{T},
    Q::AbstractMatrix{T}, R::AbstractMatrix{T}
) where {T}
    _one = one(T)
    ev(residual, x)
    objective = sum(abs2, residual)
    while true
        ev'(jacobian, x)
        copy!(Q, jacobian')
        RungeKuttaToolKit.gram_schmidt_qr!(Q, R)
        RungeKuttaToolKit.solve_lower_triangular!(residual, R)
        copy!(temp, x)
        mul!(temp, Q, residual, -_one, _one)
        ev(residual, temp)
        new_objective = sum(abs2, residual)
        if !(new_objective < objective)
            return objective
        else
            copy!(x, temp)
            objective = new_objective
        end
    end
end


ev = RKOCResidualEvaluatorBE{Float64x4}(ZHANG10_TREES, 16)
ev_full = RKOCEvaluatorBE{Float64x4}(10, 16)
x = Float64x4.(ZHANG10_LINES)
temp = Vector{Float64x4}(undef, length(x))
residual = Vector{Float64x4}(undef, length(ZHANG10_TREES))
jacobian = Matrix{Float64x4}(undef, length(ZHANG10_TREES), length(x))
Q = Matrix{Float64x4}(undef, length(x), length(ZHANG10_TREES))
R = Matrix{Float64x4}(undef, length(ZHANG10_TREES), length(ZHANG10_TREES))


coeff = Float64x4(1.0)
while true
    ev(residual, x)
    ev'(jacobian, x)
    copy!(Q, jacobian')
    RungeKuttaToolKit.gram_schmidt_qr!(Q, R)
    dx = -x
    dx -= Q * Q' * dx
    while true
        x_new = x + coeff * dx
        if (constrain!(x_new, temp, ev, residual, jacobian, Q, R) < 1.0e-100) && (ev_full(x_new) < 1.0e-100)
            copy!(x, x_new)
            coeff *= Float64x4(1.25)
            break
        else
            coeff *= Float64x4(0.5)
        end
    end
    @printf("%.20f : %.10e\n", sqrt(sum(abs2, x)), coeff)
    flush(stdout)
end
