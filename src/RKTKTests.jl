using Base: gc_alloc_count
using Test

push!(LOAD_PATH, @__DIR__)
using RKTK2
using DZMisc
using DZOptimization
using MultiprecisionFloats

################################################################################

const KNOWN_SAFE_FUNCTIONS = Regex[]
add_safe_function!(func::String) = push!(KNOWN_SAFE_FUNCTIONS,
    Regex("\"?(julia_)?" * func * "(_[0-9]+)?\"?"))

add_safe_function!("o_memset")
add_safe_function!("gemv!")
add_safe_function!("generic_matvecmul!")

is_safe(func::String)::Bool = any(
    match(safe_func, func) != nothing for safe_func in KNOWN_SAFE_FUNCTIONS)

function test_asm_calls(@nospecialize(func), @nospecialize(types))
    for call in asm_calls(func, types)
        if !is_safe(call)
            say("\nCOMPILATION TEST FAILURE:")
            say("Unsafe function call <", call,
                "> in function <", func, ">.\n")
            view_asm(func, types)
            @test false
        end
    end
end

################################################################################

using DZOptimization: update_inverse_hessian!

for T in (Float32, Float64, Float64x2, Float64x4, Float64x8)
    test_asm_calls(update_inverse_hessian!,
        (Matrix{T}, T, Vector{T}, Vector{T}, Vector{T}))
end

add_safe_function!("update_inverse_hessian!")

################################################################################

using RKTK2: compute_butcher_weights!, backprop_butcher_weights!

for T in (Float32, Float64, Float64x2, Float64x4, Float64x8)
    test_asm_calls(compute_butcher_weights!,
        (Matrix{T}, Matrix{T}, Vector{Vector{Int}}))
    test_asm_calls(backprop_butcher_weights!,
        (Matrix{T}, Matrix{T}, Vector{T}, Matrix{T}, Vector{T},
            Vector{Int}, Vector{Vector{Tuple{Int,Int}}}))
end

add_safe_function!("compute_butcher_weights!")
add_safe_function!("backprop_butcher_weights!")

################################################################################

for T in (Float32, Float64, Float64x2, Float64x4, Float64x8)
    test_asm_calls(evaluate_gradient!,
        (Matrix{T}, Vector{T}, Matrix{T}, Vector{T},
            RKOCBackpropEvaluator{T}))
end

add_safe_function!("evaluate_gradient!")

################################################################################

using DZOptimization: StepObjectiveFunctor, quadratic_line_search

for T in (Float32, Float64, Float64x2, Float64x4, Float64x8)
    test_asm_calls(quadratic_line_search,
        (StepObjectiveFunctor{RKOCExplicitBackpropObjectiveFunctor{T}, T}, T, T))
end

################################################################################

test_asm_calls(step!, (BFGSOptimizer{RKOCExplicitBackpropObjectiveFunctor{Float64}, RKOCExplicitBackpropGradientFunctor{Float64}, Float64},))
# exit()

################################################################################

function do_f64x8_ops(a::Vector{Float64x8}, b::Vector{Float64x8}, n::Int)
    @simd ivdep for i = 1 : n
        @inbounds a[i] = i
    end
    @simd ivdep for i = 1 : n
        @inbounds b[i] = sqrt(a[i])
    end
    @simd ivdep for i = 1 : n
        @inbounds a[i] = b[i] * b[i] + b[i] * b[i]
    end
    @inbounds [Float64((a[i] - 2.0 * i) + 0.0 + 0.0 + 0.0 + 0.0
                                        + 0.0 + 0.0 + 0.0 + 0.0) for i = 1 : n]
end

function test_f64x8_ops(n::Int)
    a = Vector{Float64x8}(undef, n)
    b = Vector{Float64x8}(undef, n)
    do_f64x8_ops(a, b, 10)
    result, total_time, _, gc_time, gc_diff = @timed do_f64x8_ops(a, b, n)
    for i = 1 : length(result)
        @test result[i] < 1.0e-126 * i
    end
    @test total_time < 0.5
    @test gc_time == 0.0
    @test gc_alloc_count(gc_diff) < 100
end

test_f64x8_ops(100000)

################################################################################

for T in (Float32, Float64, Float64x2, Float64x4, Float64x8)
    @test 0 == length(asm_calls(norm, (Vector{T},)))
    @test 0 == length(asm_calls(norm2, (Vector{T},)))
    @test 0 == length(asm_calls(normalize!, (Vector{T},)))
    @test 0 == length(asm_calls(approx_norm, (Vector{T},)))
    @test 0 == length(asm_calls(approx_norm2, (Vector{T},)))
    @test 0 == length(asm_calls(approx_normalize!, (Vector{T},)))
    @test 0 == length(asm_calls(dot, (Vector{T}, Vector{T})))
    @test 0 == length(asm_calls(identity_matrix!, (Matrix{T},)))
end

################################################################################

@inline function vdpol!(f, y)
    @inbounds f[1] = y[2]
    @inbounds f[2] = 1000.0 * (1.0 - y[1] * y[1]) * y[2] - y[1]
end

function run_rksolver(::Type{T}, method, n::Int) where {T <: Real}
    y0 = T[2, 0]
    h = T(2000) / T(10000000)
    sol = RKSolver{T}(method(T), 2)
    for _ = 1 : n
        runge_kutta_step!(vdpol!, y0, h, sol)
    end
    y0
end

function time_rksolver(::Type{T}, method, n::Int) where {T <: Real}
    run_rksolver(T, method, 10)
    @time run_rksolver(T, method, n)
end

function test_rksolver(::Type{T}, method, n::Int) where {T <: Real}
    run_rksolver(T, method, 10)
    _, total_time, _, gc_time, gc_diff = @timed run_rksolver(T, method, n)
    @test total_time < 0.5
    @test gc_time == 0.0
    @test gc_alloc_count(gc_diff) < 100
end

test_rksolver(Float32, rk4_table, 100000)
test_rksolver(Float64, rk4_table, 100000)
test_rksolver(Float32, rkck5_table, 100000)
test_rksolver(Float64, rkck5_table, 100000)
test_rksolver(Float32, rkf8_table, 100000)
test_rksolver(Float64, rkf8_table, 100000)

test_rksolver(Float64x2, rk4_table, 10000)
test_rksolver(Float64x3, rk4_table, 10000)
test_rksolver(Float64x4, rk4_table, 10000)
test_rksolver(Float64x5, rk4_table, 10000)
test_rksolver(Float64x6, rk4_table, 10000)
test_rksolver(Float64x7, rk4_table, 10000)
test_rksolver(Float64x8, rk4_table, 10000)

################################################################################

################################################################################

function test_backprop_evaluator()
    x = Float64x4.(rand(BigFloat, 136))
    A = zeros(Float64x4, 16, 16)
    b = zeros(Float64x4, 16)
    k = 0
    for i = 2 : 16
        for j = 1 : i-1
            A[i,j] = x[k += 1]
        end
    end
    for i = 1 : 16
        b[i] = x[k += 1]
    end

    ev1 = RKOCEvaluator{Float64x4}(10, 16)
    ev2 = RKOCBackpropEvaluator{Float64x4}(10, 16)
    residual = Vector{Float64x4}(undef, 1205)
    jacobian = Matrix{Float64x4}(undef, 1205, 136)
    evaluate_residual!(residual, x, ev1)
    evaluate_jacobian!(jacobian, x, ev1)

    gA = zeros(Float64x4, 16, 16)
    gb = zeros(Float64x4, 16)
    evaluate_gradient!(gA, gb, A, b, ev2)
    result, total_time, _, gc_time, gc_diff =
        @timed evaluate_gradient!(gA, gb, A, b, ev2)

    @test total_time < 0.05
    @test gc_time == 0.0
    @test gc_alloc_count(gc_diff) == 0
    @test Float64(result) == Float64(sum(residual.^2))

    g1 = dbl.(jacobian' * residual)
    g2 = vcat([gA[i,j] for i = 2 : 16 for j = 1 : i-1], gb)
    for (x1, x2) in zip(g1, g2)
        @test Float64(x1) == Float64(x2)
    end
end

test_backprop_evaluator()

################################################################################

for T in (Float32, Float64, Float64x2, Float64x4, Float64x8)
    @test 0 == length(asm_calls(populate_explicit!,
        (Matrix{T}, Vector{T}, Vector{T}, Int)))
    @test 0 == length(asm_calls(populate_explicit!,
        (Vector{T}, Matrix{T}, Vector{T}, Int)))
    # @test 3 == length(asm_calls(step!,
    #     (RKOCBackpropFSGDOptimizer{T}, T, Int)))
end

################################################################################

################################################################################

println("All tests passed!")
