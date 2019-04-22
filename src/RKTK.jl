println(raw"                ______ _   _______ _   __")
println(raw"                | ___ \ | / /_   _| | / /   Version  2.2")
println(raw"                | |_/ / |/ /  | | | |/ /")
println(raw"                |    /|    \  | | |    \   David K. Zhang")
println(raw"                | |\ \| |\  \ | | | |\  \     (c) 2019")
println(raw"                |_| \_\_| \_/ |_| |_| \_/")
println()
println("RKTK is free software distributed under the terms of the MIT license.")
println()
flush(stdout)

################################################################################

using Printf
using UUIDs

push!(LOAD_PATH, @__DIR__)
using DZMisc
using DZOptimization
using MultiprecisionFloats
using RKTK2

################################################################################

RKOCBFGSOptimizer{T} = BFGSOptimizer{
    RKOCExplicitBackpropObjectiveFunctor{T},
    RKOCExplicitBackpropGradientFunctor{T}, T}

struct RKTKID
    order::Int
    num_stages::Int
    uuid::UUID
end

const RKTKID_REGEX = Regex(
    "RKTK-([0-9]{2})([0-9]{2})-([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-" *
    "[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})")
const RKTK_FILENAME_REGEX = Regex(
    "^[0-9]{4}-[0-9]{4}-RKTK-([0-9]{2})([0-9]{2})-([0-9A-Fa-f]{8}-" *
    "[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})\\.txt\$")

function find_rktkid(str::String)::Union{RKTKID,Nothing}
    m = match(RKTKID_REGEX, str)
    if m != nothing
        RKTKID(parse(Int, m[1]), parse(Int, m[2]), UUID(m[3]))
    else
        nothing
    end
end

################################################################################

numstr(x)::String = @sprintf("%#-18.12g", BigFloat(x))
shortstr(x)::String = @sprintf("%#.5g", BigFloat(x))
log_score(x)::Int = round(Int, -100 * log10(BigFloat(x)))
score_str(x)::String = lpad(clamp(log_score(x), 0, 9999), 4, '0')

rktk_filename(opt, id::RKTKID)::String =
    score_str(opt.objective[1]) * '-' * score_str(norm(opt.gradient)) *
    "-RKTK-" * lpad(id.order, 2, '0') * lpad(id.num_stages, 2, '0') * '-' *
    uppercase(string(id.uuid)) * ".txt"

function round_precision(prec::Int)::Int
    if     prec <= 32;  32  # Float32
    elseif prec <= 64;  64  # Float64
    elseif prec <= 128; 128 # Float64x2
    elseif prec <= 192; 192 # Float64x3
    elseif prec <= 256; 256 # Float64x4
    elseif prec <= 320; 320 # Float64x5
    elseif prec <= 384; 384 # Float64x6
    elseif prec <= 448; 448 # Float64x7
    elseif prec <= 512; 512 # Float64x8
    else;  prec             # BigFloat
end end

function precision_type(prec::Int)::Type
    if     prec <= 32;  Float32
    elseif prec <= 64;  Float64
    elseif prec <= 128; Float64x2
    elseif prec <= 192; Float64x3
    elseif prec <= 256; Float64x4
    elseif prec <= 320; Float64x5
    elseif prec <= 384; Float64x6
    elseif prec <= 448; Float64x7
    elseif prec <= 512; Float64x8
    else;               BigFloat
end end

function print_help()::Nothing
    say("Usage: julia RKTK.jl <command> [parameters...]")
    say()
    say("RKTK provides the following <command> options:")
    say("    search <order> <num-stages> <precision>")
    say()
end

function print_table_header()::Nothing
    say(" Iteration │  Objective value  │   Gradient norm   │",
                    "  Last step size   │    Point norm     │ Type")
    say("───────────┼───────────────────┼───────────────────┼",
                    "───────────────────┼───────────────────┼──────")
end

function print_table_row(iter, obj_value, grad_norm,
                         step_size, point_norm, type)::Nothing
    say(" ", lpad(iter, 9, ' '), " | ",
        numstr(obj_value), "│ ", numstr(grad_norm),  "│ ",
        numstr(step_size), "│ ", numstr(point_norm), "│ ", type)
end

function rmk_table_row(iter, obj_value, grad_norm,
                       step_size, point_norm, type)::Nothing
    rmk(" ", lpad(iter, 9, ' '), " | ",
        numstr(obj_value), "│ ", numstr(grad_norm),  "│ ",
        numstr(step_size), "│ ", numstr(point_norm), "│ ", type)
end

function print_table_row(opt, type)::Nothing
    print_table_row(opt.iteration[1], opt.objective[1], norm(opt.gradient),
                    opt.last_step_size[1], norm(opt.current_point), type)
end

function rmk_table_row(opt, type)::Nothing
    rmk_table_row(opt.iteration[1], opt.objective[1], norm(opt.gradient),
                  opt.last_step_size[1], norm(opt.current_point), type)
end

################################################################################

function rkoc_optimizer(::Type{T}, order::Int,
        num_stages::Int) where {T <: Real}
    obj_func, grad_func = rkoc_explicit_backprop_functors(T, order, num_stages)
    num_vars = div(num_stages * (num_stages + 1), 2)
    x_init = T.(rand(BigFloat, num_vars))
    BFGSOptimizer(x_init, inv(T(1_000_000)), obj_func, grad_func)
end

function rkoc_optimizer(::Type{T}, order::Int, num_stages::Int,
        x_init::Vector{BigFloat}, num_iters::Int) where {T <: Real}
    obj_func, grad_func = rkoc_explicit_backprop_functors(T, order, num_stages)
    num_vars = div(num_stages * (num_stages + 1), 2)
    @assert length(x_init) == num_vars
    opt = BFGSOptimizer(T.(x_init), inv(T(1_000_000)), obj_func, grad_func)
    opt.iteration[1] = num_iters
    opt
end

################################################################################

function find_filename_by_id(id::RKTKID)
    for filename in readdir()
        if (m = match(RKTK_FILENAME_REGEX, filename)) != nothing
            if ((parse(Int, m[1]) == id.order) &&
                    (parse(Int, m[2]) == id.num_stages) &&
                    (UUID(m[3]) == id.uuid))
                return filename
            end
        end
    end
    return nothing
end

function save_to_file(opt::RKOCBFGSOptimizer{T}, id::RKTKID) where {T <: Real}
    filename = rktk_filename(opt, id)
    rmk("Saving progress to file ", filename, "...")
    old_filename = find_filename_by_id(id)
    if old_filename != nothing
        if old_filename != filename
            mv(old_filename, filename)
        end
        trajectory = filter(!isempty,
            strip.(split(read(filename, String), "\n\n")))
        point_data = filter(!isempty, strip.(split(trajectory[end], '\n')))
        header = split(point_data[1])
        num_iters = parse(Int, header[1])
        prec = parse(Int, header[2])
        if (precision(BigFloat) <= prec) && (opt.iteration[1] <= num_iters)
            rmk("No save necessary.")
            return
        end
    end
    file = open(filename, "a+")
    println(file, opt.iteration[1], ' ', precision(BigFloat), ' ',
            log_score(opt.objective[1]), ' ', log_score(norm(opt.gradient)))
    for x in opt.current_point
        println(file, BigFloat(x))
    end
    println(file)
    close(file)
    rmk("Save complete.")
end

function run!(opt::RKOCBFGSOptimizer{T}, id::RKTKID) where {T <: Real}
    print_table_header()
    print_table_row(opt, "NONE")
    save_to_file(opt, id)
    last_print_time = last_save_time = time_ns()
    while true
        bfgs_used, objective_decreased = step!(opt)
        if !objective_decreased
            print_table_row(opt, "DONE")
            save_to_file(opt, id)
            return
        end
        current_time = time_ns()
        if current_time - last_save_time > UInt(60_000_000_000)
            save_to_file(opt, id)
            last_save_time = current_time
        end
        if !bfgs_used
            print_table_row(opt, "GRAD")
            last_print_time = current_time
        elseif current_time - last_print_time > UInt(80_000_000)
            rmk_table_row(opt, "BFGS")
            last_print_time = current_time
        end
    end
end

function search(::Type{T}, id::RKTKID) where {T <: Real}
    optimizer = rkoc_optimizer(T, id.order, id.num_stages)
    say("Running ", T, " search RKTK-",
        lpad(id.order, 2, '0'), lpad(id.num_stages, 2, '0'), '-',
        uppercase(string(id.uuid)), ".\n")
    run!(optimizer, id)
    say("\nCompleted ", T, " search RKTK-",
        lpad(id.order, 2, '0'), lpad(id.num_stages, 2, '0'), '-',
        uppercase(string(id.uuid)), ".\n")
end

################################################################################

function refine(::Type{T}, filename::String) where {T <: Real}
    id = find_rktkid(filename)
    say("Running ", T, " refinement RKTK-",
        lpad(id.order, 2, '0'), lpad(id.num_stages, 2, '0'), '-',
        uppercase(string(id.uuid)), ".\n")
    trajectory = filter(!isempty, strip.(split(read(filename, String), "\n\n")))
    point_data = filter(!isempty, strip.(split(trajectory[end], '\n')))
    header = split(point_data[1])
    old_prec = round_precision(parse(Int, header[2]))
    if precision(BigFloat) < old_prec
        say("WARNING: Refining at lower precision than source file.\n")
    end
    optimizer = rkoc_optimizer(T, id.order, id.num_stages,
        BigFloat.(point_data[2:end]), parse(Int, header[1]))
    starting_iteration = optimizer.iteration[1]
    run!(optimizer, id)
    ending_iteration = optimizer.iteration[1]
    if ending_iteration > starting_iteration
        say("\nRepeating ", T, " refinement RKTK-",
            lpad(id.order, 2, '0'), lpad(id.num_stages, 2, '0'), '-',
            uppercase(string(id.uuid)), ".\n")
        refine(T, find_filename_by_id(id))
    else
        say("\nCompleted ", T, " refinement RKTK-",
            lpad(id.order, 2, '0'), lpad(id.num_stages, 2, '0'), '-',
            uppercase(string(id.uuid)), ".\n")
    end
end

function main()
    if (length(ARGS) == 0) || ("-h" in ARGS) || ("--help" in ARGS)
        print_help()
    elseif uppercase(ARGS[1]) == "SEARCH"
        order = parse(Int, ARGS[2])
        num_stages = parse(Int, ARGS[3])
        prec = round_precision(parse(Int, ARGS[4]))
        setprecision(prec)
        while true
            search(precision_type(prec), RKTKID(order, num_stages, uuid4()))
        end
    elseif uppercase(ARGS[1]) == "REFINE"
        id = find_rktkid(ARGS[2])
        if id == nothing
            say("ERROR: Invalid RKTK ID ", ARGS[2], ".")
            say("RKTK IDs have the form ",
                "RKTK-XXYY-ZZZZZZZZ-ZZZZ-ZZZZ-ZZZZ-ZZZZZZZZZZZZ.\n")
            exit()
        end
        filename = find_filename_by_id(id)
        if filename == nothing
            say("ERROR: No file exists with RKTK ID ", id, ".\n")
            exit()
        end
        prec = round_precision(parse(Int, ARGS[3]))
        setprecision(prec)
        refine(precision_type(prec), filename)
    elseif uppercase(ARGS[1]) == "CLEAN"
        prec = round_precision(parse(Int, ARGS[2]))
        setprecision(prec)
        for filename in reverse(readdir())
            if match(RKTK_FILENAME_REGEX, filename) != nothing
                id = find_rktkid(filename)
                refine(precision_type(prec), filename)
            end
        end
    else
        say("ERROR: Unrecognized <command> option ", ARGS[1], ".\n")
        print_help()
    end
end

main()

# const AccurateReal = Float64x4
# const THRESHOLD = AccurateReal(1e-40)

# function constrain(x::Vector{T}, evaluator::RKOCEvaluator{T}) where {T <: Real}
#     x_old, obj_old = x, norm2(evaluator(x))
#     while true
#         direction = qr(evaluator'(x_old)) \ evaluator(x_old)
#         x_new = x_old - direction
#         obj_new = norm2(evaluator(x_new))
#         if obj_new < obj_old
#             x_old, obj_old = x_new, obj_new
#         else
#             break
#         end
#     end
#     x_old, obj_old
# end

# function compute_order(x::Vector{T}, threshold::T) where {T <: Real}
#     num_stages = compute_stages(x)
#     order = 2
#     while true
#         rmk("    Testing constraints for order ", order, "...")
#         x_new, obj_new = constrain(x,
#             RKOCEvaluator{AccurateReal}(order, num_stages))
#         if obj_new <= threshold^2
#             x = x_new
#             order += 1
#         else
#             break
#         end
#     end
#     x, order - 1
# end

# function drop_last_stage(x::Vector{T}) where {T <: Real}
#     num_stages = compute_stages(x)
#     vcat(x[1 : div((num_stages - 1) * (num_stages - 2), 2)],
#          x[div(num_stages * (num_stages - 1), 2) + 1 : end - 1])
# end

# ################################################################################

# # function objective(x)
# #     x[end]^2
# # end

# # function gradient(x)
# #     result = zero(x)
# #     result[end] = dbl(x[end])
# #     result
# # end

# function constrain_step(x, step, evaluator)
#     constraint_jacobian = copy(transpose(evaluator'(x)))
#     orthonormalize_columns!(constraint_jacobian)
#     step - constraint_jacobian * (transpose(constraint_jacobian) * step)
# end

# function constrained_step_value(step_size,
#         x, step_direction, step_norm, evaluator, threshold)
#     x_new, obj_new = constrain(
#         x - (step_size / step_norm) * step_direction, evaluator)
#     if obj_new < threshold
#         objective(x_new)
#     else
#         AccurateReal(NaN)
#     end
# end

# ################################################################################

# struct ConstrainedBFGSOptimizer{T}
#     objective_value::Ref{T}
#     last_step_size::Ref{T}
# end

# ################################################################################

# if length(ARGS) != 1
#     say("Usage: julia StageReducer.jl <input-file>")
#     exit()
# end

# const INPUT_POINT = AccurateReal.(BigFloat.(split(read(ARGS[1], String))))
# say("Successfully read input file: " * ARGS[1])

# const NUM_VARS = length(INPUT_POINT)
# const NUM_STAGES = compute_stages(INPUT_POINT)
# const REFINED_POINT, ORDER = compute_order(INPUT_POINT, THRESHOLD)
# say("    ", NUM_STAGES, "-stage method of order ", ORDER,
#     " (refined by ", approx_norm(REFINED_POINT - INPUT_POINT), ").")

# const FULL_CONSTRAINTS = RKOCEvaluator{AccurateReal}(ORDER, NUM_STAGES)
# const ACTIVE_CONSTRAINT_INDICES, HI, LO = linearly_independent_column_indices!(
#     copy(transpose(FULL_CONSTRAINTS'(INPUT_POINT))), THRESHOLD)
# const ACTIVE_CONSTRAINTS = RKOCEvaluator{AccurateReal}(
#     ACTIVE_CONSTRAINT_INDICES, NUM_STAGES)
# say("    ", ACTIVE_CONSTRAINTS.num_constrs, " out of ",
#     FULL_CONSTRAINTS.num_constrs, " active constraints.")
# say("    Constraint thresholds: [",
#     shortstr(-log2(BigFloat(LO))), " | ",
#     shortstr(-log2(BigFloat(THRESHOLD))), " | ",
#     shortstr(-log2(BigFloat(HI))), "]")
# say()

# const ERROR_EVALUATOR = RKOCEvaluator{AccurateReal}(
#     Vector{Int}(rooted_tree_count(ORDER) + 1 : rooted_tree_count(ORDER + 1)),
#     NUM_STAGES)

# function objective(x)
#     norm2(ERROR_EVALUATOR(x))
# end

# function gradient(x)
#     dbl.(transpose(ERROR_EVALUATOR'(x)) * ERROR_EVALUATOR(x))
# end

# const OPT = ConstrainedBFGSOptimizer{AccurateReal}(
#     objective(REFINED_POINT),
#     AccurateReal(0.00001))




# function main()
#     x = copy(REFINED_POINT)
#     inv_hess = Matrix{AccurateReal}(I, NUM_VARS, NUM_VARS)

#     cons_grad = constrain_step(x, gradient(x), ACTIVE_CONSTRAINTS)
#     cons_grad_norm = norm(cons_grad)
#     print_table_header()
#     print_table_row(OPT.objective_value[], cons_grad_norm, 0, "NONE")

#     while true

#         rmk("Performing gradient descent step...")
#         grad_step_size, obj_grad = quadratic_search(constrained_step_value,
#             OPT.last_step_size[], x, cons_grad, cons_grad_norm,
#             ACTIVE_CONSTRAINTS, THRESHOLD)

#         rmk("Performing BFGS step...")
#         bfgs_step = constrain_step(x, inv_hess * cons_grad, ACTIVE_CONSTRAINTS)
#         bfgs_step_norm = norm(bfgs_step)
#         bfgs_step_size, obj_bfgs = quadratic_search(constrained_step_value,
#             OPT.last_step_size[], x, bfgs_step, bfgs_step_norm,
#             ACTIVE_CONSTRAINTS, THRESHOLD)

#         rmk("Line searches complete.")
#         if obj_bfgs < OPT.objective_value[] && obj_bfgs <= obj_grad
#             x, _ = constrain(x - (bfgs_step_size / bfgs_step_norm) * bfgs_step,
#             ACTIVE_CONSTRAINTS)
#             cons_grad_new = constrain_step(x, gradient(x), ACTIVE_CONSTRAINTS)
#             cons_grad_norm_new = norm(cons_grad_new)
#             update_inverse_hessian!(inv_hess,
#                 -bfgs_step_size / bfgs_step_norm,
#                 bfgs_step,
#                 cons_grad_new - cons_grad,
#                 Vector{AccurateReal}(undef, NUM_VARS))
#             OPT.last_step_size[] = bfgs_step_size
#             OPT.objective_value[] = obj_bfgs
#             cons_grad = cons_grad_new
#             cons_grad_norm = cons_grad_norm_new
#             print_table_row(OPT.objective_value[], cons_grad_norm,
#                 OPT.last_step_size[], "")
#         elseif obj_grad < OPT.objective_value[]
#             x, _ = constrain(x - (grad_step_size / cons_grad_norm) * cons_grad,
#                 ACTIVE_CONSTRAINTS)
#             inv_hess = Matrix{AccurateReal}(I, NUM_VARS, NUM_VARS)
#             OPT.last_step_size[] = grad_step_size
#             OPT.objective_value[] = obj_grad
#             cons_grad = constrain_step(x, gradient(x), ACTIVE_CONSTRAINTS)
#             cons_grad_norm = norm(cons_grad)
#             print_table_row(OPT.objective_value[], cons_grad_norm,
#                 OPT.last_step_size[], "GRAD")
#         else
#             print_table_row(OPT.objective_value[], cons_grad_norm, 0, "DONE")
#             say()
#             break
#         end

#     end

#     println.(string.(BigFloat.(x)))

#     # x_new = drop_last_stage(x)
#     # println.(string.(BigFloat.(x_new)))
# end

# main()
