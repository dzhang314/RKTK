using Distributed: @distributed, @everywhere, pmap

@everywhere begin
    using Dates: now, format, @dateformat_str
    using LinearAlgebra: dot, norm
    using Printf: @sprintf
    push!(LOAD_PATH, @__DIR__)
    using GoldenSectionSearch: golden_section_search
    using DZMisc: log, rooted_tree_count, orthonormalize_columns
end # @everywhere

# # Vector projection of a onto b
# proj(a, b) = (dot(b, a) / dot(b, b)) * b

# function linearly_independent_columns(mat::Matrix{T}) where {T}
#     # TODO: This assumes the first column of mat is nonzero.
#     vecs = Vector{T}[mat[:,1]]
#     idxs = Int[1]
#     root_eps = sqrt(eps(T))
#     threshold = root_eps * sqrt(root_eps)
#     for i = 2 : size(mat, 2)
#         next = mat[:,i]
#         for v in vecs
#             next -= proj(next, v)
#         end
#         if norm(next) > threshold
#             push!(idxs, i)
#             push!(vecs, next)
#         end
#     end
#     hcat(vecs...), idxs
# end

@everywhere begin

    const ORDER = 10
    const NUM_STAGES = 16
    const NUM_CONSTRS = convert(Int, rooted_tree_count(ORDER))
    const NUM_VARS = div(NUM_STAGES * (NUM_STAGES + 1), 2)
    const RK_PREC = 192

    Base.MPFR.setprecision(RK_PREC)
    const RKEVAL_CMD = (Sys.iswindows()
        ? `C:\\Programs\\rkeval.exe $(ORDER) $(NUM_STAGES) $(RK_PREC) -`
        : `/opt/rkeval $(ORDER) $(NUM_STAGES) $(RK_PREC) -`)
    const RKEVAL_PROC = open(RKEVAL_CMD, read=true, write=true)

    function rkeval_write(RKEVAL_PROC, xs, code)
        for x in xs
            write(RKEVAL_PROC, string(x))
            write(RKEVAL_PROC, '\n')
        end
        write(RKEVAL_PROC, code)
        flush(RKEVAL_PROC)
    end

    function rkobj(x)
        rkeval_write(RKEVAL_PROC, x, 'A')
        BigFloat(readline(RKEVAL_PROC))
    end

    function rkgrad(x)
        rkeval_write(RKEVAL_PROC, x, 'B')
        [BigFloat(readline(RKEVAL_PROC)) for _ = 1 : NUM_VARS]
    end

    function rkgrad!(g, x)
        rkeval_write(RKEVAL_PROC, x, 'B')
        for i = 1 : NUM_VARS
            g[i] = BigFloat(readline(RKEVAL_PROC))
        end
    end

    function rkres(x)
        rkeval_write(RKEVAL_PROC, x, 'C')
        [BigFloat(readline(RKEVAL_PROC)) for _ = 1 : NUM_CONSTRS]
    end

    function rkjac(x)
        rkeval_write(RKEVAL_PROC, x, 'D')
        reshape([BigFloat(readline(RKEVAL_PROC))
                    for _ = 1 : NUM_CONSTRS * NUM_VARS],
                NUM_CONSTRS, NUM_VARS)
    end

end # @everywhere

# const INIT_DATA = BigFloat.(INIT_DATA_STR)
# @assert all(rkres(INIT_DATA) .< 16 * eps(BigFloat))

@everywhere const CONSTR_IDXS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 31, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52,
    58, 59, 62, 75, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 106, 107, 110, 115, 116, 123, 134, 135, 201, 202, 203, 205, 206,
    207, 208, 210, 211, 214, 215, 221, 222, 225, 226, 227, 230, 231, 238, 249,
    250, 269, 270, 316, 404, 406, 487, 488, 489, 490, 491, 492, 493, 496, 497,
    498, 499, 500, 501, 502, 507, 508, 511, 516, 517, 524, 535, 536, 539, 544,
    555, 556, 602, 603, 650, 690, 984]

################################################################################

@everywhere function jacobian_projection(x, vec)
    jac = transpose(rkjac(x))[:,CONSTR_IDXS]
    orthonormalize_columns(jac)
    vec - jac * (transpose(jac) * vec)
end

@everywhere function force_energy(points, i)
    # log("    Computing force on point ", i, ".")
    force = zeros(BigFloat, NUM_VARS)
    energy = zero(BigFloat)
    for j = 1 : length(points)
        if i != j
            displ = points[i] - points[j]
            inv_dist = inv(norm(displ))
            force += inv_dist^3 * displ
            energy += inv_dist
        end
    end
    # log("    Computing Jacobian projection for point ", i, ".")
    jacobian_projection(points[i], force), energy
end

@everywhere function constrain(x)
    x_old, obj_old = x, rkobj(x)
    while true
        # log("        Computing Jacobian and residual...")
        res = rkres(x_old)[CONSTR_IDXS]
        jac = rkjac(x_old)[CONSTR_IDXS,:]
        # log("        Computing step direction...")
        direction = jac \ -res
        x_new = x_old + direction
        obj_new = rkobj(x_new)
        if 100 * obj_new < obj_old
            # log("        Accepted step: ", @sprintf("%g", obj_new))
            x_old, obj_old = x_new, obj_new
        else
            # log("        Rejected step: ", @sprintf("%g", obj_new))
            step_size, obj_new = golden_section_search(
                h -> rkobj(x_new + h * direction), 0, 1, 10)
            x_new = x_old + step_size * direction
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

@everywhere const EPS_THREE_HALVES = eps(BigFloat) * sqrt(eps(BigFloat))

@everywhere function perturb(x, direction, multiplier)
    x_new, obj = constrain(x + multiplier * direction)
    if obj < EPS_THREE_HALVES
        x_new
    else
        multiplier /= 2
        log("WARNING: Lowering perturbation step size to ", multiplier, ".")
        # x_new = perturb(x, direction, multiplier)
        # perturb(x_new, direction, multiplier)
        perturb(x, direction, multiplier)
    end
end

@everywhere function perturb_wrapper(x, direction, i)
    log("    Moving point ", i, " by distance ",
        @sprintf("%g", norm(direction)), ".")
    x_new = perturb(x, direction, 1.0)
    log("    Successfully moved point ", i, " by distance ",
        @sprintf("%g", norm(x_new - x)), ".")
    x_new
end

const INPUT_FILENAME = maximum(filename
    for filename in readdir()
    if isfile(filename) && startswith(filename, "RKTK-POINTS-")
                        && endswith(filename, ".txt"))

log("Reading initial points from data file: ", INPUT_FILENAME)
const POINTS = [BigFloat.(point)
    for point in split.(split(read(INPUT_FILENAME, String), "\n\n"))]

@assert all(length(p) == NUM_VARS for p in POINTS)

while true
    log("Computing forces...")
    forces_energies = pmap(force_energy, [POINTS for _ = 1 : length(POINTS)], 1 : length(POINTS))
    forces = [p[1] for p in forces_energies]
    total_energy = sum(p[2] for p in forces_energies) / 2
    log("Total energy: ", total_energy)
    log("Moving points...")
    forces *= sqrt(big(length(POINTS))) / norm(vcat(forces...)) / 3000
    new_points = pmap(perturb_wrapper, POINTS, forces, 1 : length(POINTS))
    for i = 1 : length(POINTS)
        POINTS[i] = new_points[i]
    end
    filename = format(now(), dateformat"RKTK-POINT\S-yyyymmdd-HHMMSS-sss.txt")
    log("Writing points to file: ", filename)
    write(filename, join([join(string.(p), "\n") for p in POINTS], "\n\n") * "\n")
end
