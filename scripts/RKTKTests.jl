using Base: gc_alloc_count
using Test

push!(LOAD_PATH, @__DIR__)
using RKTK2
using MultiprecisionFloats

################################################################################

function warmup_rkocevaluator()
    evaluator = RKOCEvaluator{Float64}(4, 4)
    for residual in evaluator(rk4_table(Float64))
        @test abs(residual) < 1.0e-16
    end
end

warmup_rkocevaluator()

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
test_rksolver(Float32, rkfnc_table, 100000)
test_rksolver(Float64, rkfnc_table, 100000)
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

println("All tests passed!")
