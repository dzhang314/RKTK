using LinearAlgebra: qr!
using Dates: @dateformat_str, now, format
using Printf: @sprintf

push!(LOAD_PATH, @__DIR__)
using MultiprecisionFloats
using RKTK2
using DZMisc: say, approx_norm, orthonormalize_columns!

const AccurateReal = Float64x2
setprecision(256)

################################################################################

const EVALUATOR = RKOCEvaluator{AccurateReal}(10, 16)
const ERROR_EVALUATOR = RKOCEvaluator{AccurateReal}(11, 16)
const NUM_VARS = 136
const NUM_CONSTRS = 1205

const CONSTR_IDXS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 31, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52,
    58, 59, 62, 75, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 106, 107, 110, 115, 116, 123, 134, 135, 201, 202, 203, 205, 206,
    207, 208, 210, 211, 214, 215, 221, 222, 225, 226, 227, 230, 231, 238, 249,
    250, 269, 270, 316, 404, 406, 487, 488, 489, 490, 491, 492, 493, 496, 497,
    498, 499, 500, 501, 502, 507, 508, 511, 516, 517, 524, 535, 536, 539, 544,
    555, 556, 602, 603, 650, 690, 984]

const CONSTR_DIM = length(CONSTR_IDXS)

const TEMP_RES = Vector{AccurateReal}(undef, NUM_CONSTRS)
const SHORT_RES = Vector{AccurateReal}(undef, CONSTR_DIM)
const TEMP_JAC = Matrix{AccurateReal}(undef, NUM_CONSTRS, NUM_VARS)
const SHORT_JAC = Matrix{AccurateReal}(undef, NUM_VARS, CONSTR_DIM)
const SHORT_JCT = Matrix{AccurateReal}(undef, CONSTR_DIM, NUM_VARS)
const APX_SHORT_JAC = Matrix{Float64}(undef, NUM_VARS, CONSTR_DIM)
const ERROR_COEFFS = Vector{AccurateReal}(undef, ERROR_EVALUATOR.num_constrs)
const ERROR_JAC = Matrix{AccurateReal}(undef, ERROR_EVALUATOR.num_constrs, NUM_VARS)

################################################################################

function approx_objective(x)
    evaluate_residual!(TEMP_RES, x, EVALUATOR)
    result = zero(Float64)
    for i = 1 : NUM_CONSTRS
        result += abs2(Float64(TEMP_RES[i]))
    end
    result
end

function compute_residual(x)
    evaluate_residual!(TEMP_RES, x, EVALUATOR)
    @simd ivdep for j = 1 : length(CONSTR_IDXS)
        @inbounds SHORT_RES[j] = TEMP_RES[CONSTR_IDXS[j]]
    end
end

function compute_jacobian(x)
    evaluate_jacobian!(TEMP_JAC, x, EVALUATOR)
    for j = 1 : length(CONSTR_IDXS)
        k = CONSTR_IDXS[j]
        @simd ivdep for i = 1 : NUM_VARS
            @inbounds SHORT_JCT[j, i] = TEMP_JAC[k, i]
        end
    end
end

function compute_orthonormalized_jacobian(x)
    evaluate_jacobian!(TEMP_JAC, x, EVALUATOR)
    for j = 1 : length(CONSTR_IDXS)
        k = CONSTR_IDXS[j]
        @simd ivdep for i = 1 : NUM_VARS
            @inbounds SHORT_JAC[i, j] = TEMP_JAC[k, i]
        end
    end
    orthonormalize_columns!(SHORT_JAC)
    for j = 1 : length(CONSTR_IDXS)
        @simd ivdep for i = 1 : NUM_VARS
            @inbounds APX_SHORT_JAC[i, j] = Float64(SHORT_JAC[i, j])
        end
    end
end

function compute_error_coefficients(point)
    evaluate_error_coefficients!(ERROR_COEFFS, point, ERROR_EVALUATOR)
end

function compute_error_jacobian(point)
    evaluate_error_jacobian!(ERROR_JAC, point, ERROR_EVALUATOR)
end

function approx_force(point::Vector{T}) where {T <: Real}
    compute_error_coefficients(point)
    compute_error_jacobian(point)
    force = transpose(ERROR_JAC[1206:end, :]) * ERROR_COEFFS[1206:end]
    compute_orthonormalized_jacobian(point)
    force - APX_SHORT_JAC * (transpose(APX_SHORT_JAC) * force)
end

################################################################################

function constrain(x)
    x_old, obj_old = x, approx_objective(x)
    while true
        compute_residual(x_old)
        compute_jacobian(x_old)
        direction = qr!(SHORT_JCT) \ SHORT_RES
        x_new = x_old - direction
        obj_new = approx_objective(x_new)
        if obj_new < obj_old
            x_old, obj_old = x_new, obj_new
        else
            break
        end
    end
    x_old, obj_old
end

const EPS_THREE_HALVES = Float64(BigFloat(2)^-130)

function perturb(x, direction, multiplier)
    x_new, obj = constrain(x + multiplier * direction)
    if obj < EPS_THREE_HALVES
        x_new, multiplier
    else
        multiplier /= 2
        perturb(x, direction, multiplier)
    end
end

################################################################################

const INPUT_FILENAME = maximum(filename
    for filename in readdir()
    if isfile(filename) && startswith(filename, "RKTK-ERROPT-")
                        && endswith(filename, ".txt"))

say("Reading initial point from data file: ", INPUT_FILENAME)
const INPUT_POINT = AccurateReal.(BigFloat.(split(read(INPUT_FILENAME, String))))

@assert length(INPUT_POINT) == NUM_VARS
say("Successfully read initial point.")

function main()
    x = INPUT_POINT[:]
    compute_error_coefficients(x)
    old_norm = approx_norm(ERROR_COEFFS[1206:end])
    println(old_norm)
    speed = 2500.0
    while true
        for _ = 1 : 1000
            force = approx_force(x)
            while true
                force *= inv(-speed * approx_norm(force))
                x_new, multiplier = perturb(x, force, 1.0)
                compute_error_coefficients(x_new)
                new_norm = approx_norm(ERROR_COEFFS[1206:end])
                if new_norm < old_norm
                    x = x_new
                    old_norm = new_norm
                    if multiplier == 1.0
                        speed *= 0.95
                    else
                        speed *= 2.0
                    end
                    println(new_norm, " ", speed, " ", multiplier)
                    break
                else
                    speed *= 2.0
                end
            end
        end
        filename = format(now(), dateformat"RKTK-\ERROPT-yyyymmdd-HHMMSS-sss.txt")
        say("Writing points to file: ", filename)
        write(filename, join(string.(BigFloat.(x)), "\n") * "\n")
    end
end

main()
