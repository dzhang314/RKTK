using Base: gc_alloc_count
using Test

using AssemblyView
using DZLinearAlgebra
using DZOptimization
using MultiFloats
using RungeKuttaToolKit

function rmk(args...)::Nothing
    print("\33[2K\r")
    print(args...)
    flush(stdout)
end

function say(args...)::Nothing
    print("\33[2K\r")
    println(args...)
    flush(stdout)
end

################################################################################

const KNOWN_SAFE_FUNCTIONS = Regex[]

add_safe_function!(func::String) = push!(KNOWN_SAFE_FUNCTIONS,
    Regex("\"?(julia_)?" * func * "(_[0-9]+)?\"?"))

add_safe_function!("o_memset")
add_safe_function!("gemv!")
add_safe_function!("generic_matvecmul!")

is_safe(func::String)::Bool = any(
    match(safe_func, func) != nothing for safe_func in KNOWN_SAFE_FUNCTIONS)

replace_underscores(expr::Expr, t::Symbol) = Expr(
    replace_underscores(expr.head, t),
    [replace_underscores(arg, t) for arg in expr.args]...)

replace_underscores(sym::Symbol, t::Symbol) = (sym == :_) ? t : sym

replace_underscores(qn::QuoteNode, t::Symbol) =
    QuoteNode(replace_underscores(qn.value, t))

function test_asm_calls(@nospecialize(func), type_template::Expr)
    for T in (:Float32, :Float64, :Float64x2, :Float64x4, :Float64x8)
        rmk("    ", rpad(func, 40, ' '), ": ", T)
        types = eval(replace_underscores(type_template, T))
        for call in asm_offsets(func, types)
            if !is_safe(call)
                say("\nCOMPILATION TEST FAILURE:")
                say("Unsafe function call <", call,
                    "> in function <", func, ">.\n")
                exit()
            end
        end
    end
    say("    ", rpad(func, 40, ' '), ": PASS")
end

macro test_block(name::String, block::Expr)
    Expr(:block, :(say("Running ", $name, " tests...")),
        block, :(say("All ", $name, " tests passed.\n")))
end

say()

################################################################################

@test_block "DZMisc compilation" begin
    test_asm_calls(norm, :(Vector{_},))
    test_asm_calls(norm2, :(Vector{_},))
    test_asm_calls(normalize!, :(Vector{_},))
    test_asm_calls(dot, :(Vector{_}, Vector{_}))
    test_asm_calls(identity_matrix!, :(Matrix{_},))
end

@test_block "RKTK2 compilation" begin
    test_asm_calls(populate_explicit!, :(Matrix{_}, Vector{_}, Vector{_}, Int))
    test_asm_calls(populate_explicit!, :(Vector{_}, Matrix{_}, Vector{_}, Int))
    add_safe_function!("populate_explicit!")
    test_asm_calls(RungeKuttaToolKit.compute_butcher_weights!,
        :(Matrix{_}, Matrix{_}, Vector{Pair{Int,Int}}))
    add_safe_function!("jl_throw")
    add_safe_function!("jl_system_image_data")
    test_asm_calls(RungeKuttaToolKit.backprop_butcher_weights!,
        :(Matrix{_}, Matrix{_}, Vector{_}, Matrix{_}, Vector{_},
            Vector{Int}, Vector{Vector{Tuple{Int,Int}}}))
    add_safe_function!("compute_butcher_weights!")
    add_safe_function!("backprop_butcher_weights!")
    test_asm_calls(evaluate_residual2,
        :(Matrix{_}, Vector{_}, RKOCBackpropEvaluator{_}))
    test_asm_calls(evaluate_gradient!,
        :(Matrix{_}, Vector{_}, Matrix{_}, Vector{_}, RKOCBackpropEvaluator{_}))
    add_safe_function!("evaluate_residual2")
    add_safe_function!("evaluate_gradient!")
end

@test_block "DZOptimization compilation" begin
    test_asm_calls(DZOptimization.update_inverse_hessian!,
        :(Matrix{_}, _, Vector{_}, Vector{_}, Vector{_}))
    test_asm_calls(DZOptimization.quadratic_line_search,
        :(DZOptimization.StepObjectiveFunctor{
            RKOCExplicitBackpropObjectiveFunctor{_}, _}, _, _))
    add_safe_function!("update_inverse_hessian!")
    add_safe_function!("quadratic_line_search")
    test_asm_calls(step!,
        :(BFGSOptimizer{RKOCExplicitBackpropObjectiveFunctor{_},
            RKOCExplicitBackpropGradientFunctor{_}, _},))
    add_safe_function!("step!")
end

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
    @inbounds [Float64(abs(a[i] - 2.0 * i)) for i = 1 : n]
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

@test_block "Float64x8 performance" begin
    test_f64x8_ops(100000)
end

################################################################################

@inline function vdpol!(f, y)
    @inbounds f[1] = y[2]
    @inbounds f[2] = 1000.0 * (1.0 - y[1] * y[1]) * y[2] - y[1]
end

function run_rksolver(::Type{T}, method, n::Int) where {T <: Real}
    y0 = T[2, 0]
    h = T(2000) / T(10000000)
    sol = RungeKuttaToolKit.RKSolver{T}(method(T), 2)
    for _ = 1 : n
        RungeKuttaToolKit.runge_kutta_step!(vdpol!, y0, h, sol)
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

using RungeKuttaToolKit.ExampleMethods
@test_block "RKSolver performance" begin
    test_rksolver(Float32, rk4_table,   100000)
    test_rksolver(Float64, rk4_table,   100000)
    test_rksolver(Float32, rkck5_table, 100000)
    test_rksolver(Float64, rkck5_table, 100000)
    test_rksolver(Float32, rkf8_table,  100000)
    test_rksolver(Float64, rkf8_table,  100000)
    test_rksolver(Float64x2, rk4_table, 10000)
    test_rksolver(Float64x3, rk4_table, 10000)
    test_rksolver(Float64x4, rk4_table, 10000)
    test_rksolver(Float64x5, rk4_table, 10000)
    test_rksolver(Float64x6, rk4_table, 10000)
    test_rksolver(Float64x7, rk4_table, 10000)
    test_rksolver(Float64x8, rk4_table, 10000)
end

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

    g1 = jacobian' * residual
    g1 += g1
    g2 = vcat([gA[i,j] for i = 2 : 16 for j = 1 : i-1], gb)
    for (x1, x2) in zip(g1, g2)
        @test Float64(x1) == Float64(x2)
    end
end

@test_block "Backprop evaluator" begin
    test_backprop_evaluator()
end
