using Distributed: @everywhere, pmap

@everywhere begin
    using LinearAlgebra: normalize!
    using Dates: @dateformat_str, now, format
    using Printf: @sprintf
    using SharedArrays

    push!(LOAD_PATH, @__DIR__)
    using QuadFloat
    using RKTK
    using DZMisc: log, orthonormalize_columns!
    using GoldenSectionSearch: golden_section_search

    import Base: ==, *
    ==(x::TripleF64, y::Int) = (x.x0 == y)
    *(x::Int, y::TripleF64) = Float64(x) * y
    setprecision(180)
end

@everywhere function approximate_norm(x::Vector{T}) where {T <: Real}
    result = 0.0
    for i = 1 : length(x)
        result += abs2(Float64(x[i]))
    end
    sqrt(result)
end

@everywhere function approximate_coulomb_force_energy(
        points::Matrix{T}, i::Int) where {T <: Real}
    displ = Vector{Float64}(undef, size(points, 1))
    force = zeros(Float64, size(points, 1))
    energy = zero(Float64)
    for j = 1 : size(points, 2)
        if i != j
            @simd ivdep for k = 1 : size(points, 1)
                @inbounds displ[k] = (
                    Float64(points[k,i]) - Float64(points[k,j]))
            end
            inv_dist = inv(approximate_norm(displ))
            energy += inv_dist
            inv_dist *= abs2(inv_dist)
            @simd ivdep for k = 1 : size(points, 1)
                @inbounds force[k] += inv_dist * displ[k]
            end
        end
    end
    force, energy
end

@everywhere function approximate_coulomb_force_energy(
        points::SharedArray{T,2}, i::Int) where {T <: Real}
    force, energy = approximate_coulomb_force_energy(sdata(points), i)
    compute_orthonormalized_jacobian(points[:,i])
    force - APX_SHORT_JAC * (transpose(APX_SHORT_JAC) * force), energy
end

################################################################################

@everywhere const EVALUATOR = RKOCEvaluator{TripleF64}(10, 16)
@everywhere const NUM_VARS = 136
@everywhere const NUM_CONSTRS = 1205

@everywhere const CONSTR_IDXS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 31, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52,
    58, 59, 62, 75, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 106, 107, 110, 115, 116, 123, 134, 135, 201, 202, 203, 205, 206,
    207, 208, 210, 211, 214, 215, 221, 222, 225, 226, 227, 230, 231, 238, 249,
    250, 269, 270, 316, 404, 406, 487, 488, 489, 490, 491, 492, 493, 496, 497,
    498, 499, 500, 501, 502, 507, 508, 511, 516, 517, 524, 535, 536, 539, 544,
    555, 556, 602, 603, 650, 690, 984]

@everywhere const CONSTR_DIM = length(CONSTR_IDXS)

@everywhere const TEMP_RES = Vector{TripleF64}(undef, NUM_CONSTRS)
@everywhere const SHORT_RES = Vector{TripleF64}(undef, CONSTR_DIM)
@everywhere const APX_SHORT_RES = Vector{Float64}(undef, CONSTR_DIM)
@everywhere const TEMP_JAC = Matrix{TripleF64}(undef, NUM_CONSTRS, NUM_VARS)
@everywhere const SHORT_JAC = Matrix{TripleF64}(undef, NUM_VARS, CONSTR_DIM)
@everywhere const APX_SHORT_JAC = Matrix{Float64}(undef, NUM_VARS, CONSTR_DIM)

################################################################################

@everywhere function approximate_objective(x)
    evaluate_residuals!(TEMP_RES, x, EVALUATOR)
    result = zero(Float64)
    for i = 1 : NUM_CONSTRS
        result += abs2(Float64(TEMP_RES[i]))
    end
    result
end

@everywhere function compute_residuals(x)
    evaluate_residuals!(TEMP_RES, x, EVALUATOR)
    for j = 1 : length(CONSTR_IDXS)
        SHORT_RES[j] = TEMP_RES[CONSTR_IDXS[j]]
    end
end

@everywhere function compute_jacobian(x)
    evaluate_jacobian!(TEMP_JAC, x, EVALUATOR)
    for j = 1 : length(CONSTR_IDXS)
        k = CONSTR_IDXS[j]
        for i = 1 : NUM_VARS
            SHORT_JAC[i, j] = TEMP_JAC[k, i]
        end
    end
end

@everywhere function compute_orthonormalized_jacobian(x)
    evaluate_jacobian!(TEMP_JAC, x, EVALUATOR)
    for j = 1 : length(CONSTR_IDXS)
        k = CONSTR_IDXS[j]
        for i = 1 : NUM_VARS
            SHORT_JAC[i, j] = TEMP_JAC[k, i]
        end
    end
    orthonormalize_columns!(SHORT_JAC)
    for j = 1 : length(CONSTR_IDXS)
        for i = 1 : NUM_VARS
            APX_SHORT_JAC[i, j] = Float64(SHORT_JAC[i, j])
        end
    end
end

################################################################################

const INPUT_FILENAME = maximum(filename
    for filename in readdir()
    if isfile(filename) && startswith(filename, "RKTK-POINTS-")
                        && endswith(filename, ".txt"))

log("Reading initial points from data file: ", INPUT_FILENAME)
const INPUT_POINTS = [TripleF64.(BigFloat.(point))
    for point in split.(split(read(INPUT_FILENAME, String), "\n\n"))]
@assert all(length(p) == NUM_VARS for p in INPUT_POINTS)
const NUM_POINTS = length(INPUT_POINTS)

const POINTS = SharedArray{TripleF64}((NUM_VARS, NUM_POINTS))
for j = 1 : NUM_POINTS
    pt = INPUT_POINTS[j]
    for i = 1 : NUM_VARS
        POINTS[i, j] = pt[i]
    end
end
log("Successfully read initial points.")

################################################################################

@everywhere function constrain(x)
    x_old, obj_old = x, approximate_objective(x)
    while true
        # log("        Computing Jacobian and residual...")
        compute_residuals(x_old)
        compute_jacobian(x_old)
        # log("        Computing step direction...")
        direction = transpose(SHORT_JAC) \ SHORT_RES
        x_new = x_old - direction
        obj_new = approximate_objective(x_new)
        if 100 * obj_new < obj_old
            # log("        Accepted step: ", @sprintf("%g", obj_new))
            x_old, obj_old = x_new, obj_new
        else
            # log("        Rejected step: ", @sprintf("%g", obj_new))
            step_size, obj_new = golden_section_search(
                h -> approximate_objective(x_new - h * direction), 0, 1, 10)
            x_new = x_old - step_size * direction
            if 2 * obj_new < obj_old
                # log("        Accepted GSS step: ", @sprintf("%g", obj_new),
                #     " (step size ", @sprintf("%g", step_size), ")")
                x_old, obj_old = x_new, obj_new
            else
                # log("        Rejected GSS step: ", @sprintf("%g", obj_new))
                break
            end
        end
    end
    x_old, obj_old
end

@everywhere const EPS_THREE_HALVES = Float64(BigFloat(2)^-200)

@everywhere function perturb(x, direction, multiplier, i)
    x_new, obj = constrain(x + multiplier * direction)
    if obj < EPS_THREE_HALVES
        x_new
    else
        multiplier /= 2
        if multiplier < 0.5
            log("WARNING: Lowering perturbation step size for point ", i,
                " to ", multiplier, " * ", approximate_norm(direction), ".")
        end
        # x_new = perturb(x, direction, multiplier)
        # perturb(x_new, direction, multiplier)
        perturb(x, direction, multiplier, i)
    end
end

@everywhere function perturb_wrapper(pts, direction, i)
    x = sdata(pts)[:,i]
    # log("    Moving point ", i, " by distance ",
    #     @sprintf("%g", approximate_norm(direction)), ".")
    x_new = perturb(x, direction, 1.0, i)
    # log("    Successfully moved point ", i, " by distance ",
    #     @sprintf("%g", approximate_norm(x_new - x)), ".")
    x_new
end

################################################################################

while true
    # log("Computing forces...")
    forces_energies = pmap(approximate_coulomb_force_energy,
        [POINTS for _ = 1 : NUM_POINTS], 1 : NUM_POINTS)
    log("Total energy: ", sum(p[2] for p in forces_energies) / 2)
    forces = [p[1] for p in forces_energies]
    # log("Moving points...")
    forces *= sqrt(NUM_POINTS) / approximate_norm(vcat(forces...)) / 5000
    new_points = pmap(perturb_wrapper,
        [POINTS for _ = 1 : NUM_POINTS], forces, 1 : NUM_POINTS)
    for i = 1 : NUM_POINTS
        for j = 1 : NUM_VARS
            POINTS[j, i] = new_points[i][j]
        end
    end
    filename = format(now(), dateformat"RKTK-POINT\S-yyyymmdd-HHMMSS-sss.txt")
    log("Writing points to file: ", filename)
    write(filename, join([join(string.(BigFloat.(exact.(POINTS[:,i]))), "\n") for i = 1 : NUM_POINTS], "\n\n") * "\n")
end
