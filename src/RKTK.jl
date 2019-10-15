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

using Base.Threads: @threads, nthreads, threadid
using Printf: @sprintf
using Statistics: mean, std
using UUIDs: UUID, uuid4

using MultiFloats
using RungeKuttaToolKit

push!(LOAD_PATH, @__DIR__)
using DZMisc
using DZOptimization

################################################################################

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
    else
        setprecision(prec)
        BigFloat
    end
end

approx_precision(::Type{Float32  }) = 32
approx_precision(::Type{Float64  }) = 64
approx_precision(::Type{Float64x2}) = 128
approx_precision(::Type{Float64x3}) = 192
approx_precision(::Type{Float64x4}) = 256
approx_precision(::Type{Float64x5}) = 320
approx_precision(::Type{Float64x6}) = 384
approx_precision(::Type{Float64x7}) = 448
approx_precision(::Type{Float64x8}) = 512
approx_precision(::Type{BigFloat }) = precision(BigFloat)

function Base.show(io::IO, ::Type{MultiFloat{Float64,N}}) where {N}
    write(io, "Float64x")
    show(io, N)
end

function Base.show(io::IO, ::Type{BigFloat})
    write(io, "BigFloat(")
    show(io, precision(BigFloat))
    write(io, ')')
end

################################################################################

RKOCBFGSOptimizer{T} = BFGSOptimizer{
    RKOCExplicitBackpropObjectiveFunctor{T},
    RKOCExplicitBackpropGradientFunctor{T}, T}

struct RKTKID
    order::Int
    num_stages::Int
    uuid::UUID
end

function Base.show(io::IO, id::RKTKID)
    write(io, "RKTK-")
    write(io, lpad(id.order, 2, '0'))
    write(io, lpad(id.num_stages, 2, '0'))
    write(io, '-')
    write(io, uppercase(string(id.uuid)))
end

const RKTKID_REGEX = Regex(
    "RKTK-([0-9]{2})([0-9]{2})-([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-" *
    "[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})")
const RKTK_FILENAME_REGEX = Regex(
    "^[0-9]{4}-[0-9]{4}-[0-9]{4}-RKTK-([0-9]{2})([0-9]{2})-([0-9A-Fa-f]{8}-" *
    "[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})\\.txt\$")

function find_rktkid(str::String)::Union{RKTKID,Nothing}
    m = match(RKTKID_REGEX, str)
    if m != nothing
        RKTKID(parse(Int, m[1]), parse(Int, m[2]), UUID(m[3]))
    else
        nothing
    end
end

function find_filename_by_id(dir::String, id::RKTKID)::Union{String,Nothing}
    result = nothing
    for filename in readdir(dir)
        m = match(RKTKID_REGEX, filename)
        if ((m != nothing) && (parse(Int, m[1]) == id.order) &&
                (parse(Int, m[2]) == id.num_stages) && (UUID(m[3]) == id.uuid))
            if result != nothing
                say("ERROR: Found multiple files with RKTK ID $id.")
                exit()
            else
                result = filename
            end
        end
    end
    result
end

################################################################################

function log_score(x)::Int
    bx = BigFloat(x)
    if     iszero(bx)  ; typemax(Int)
    elseif isfinite(bx); round(Int, -100 * log10(bx))
    else               ; 0
    end
end

function scaled_log_score(x)
    bx = BigFloat(x)
    if     iszero(bx)  ; typemax(Int)
    elseif isfinite(bx); round(Int, 10000 - 3333 * log10(bx) / 2)
    else               ; 0
    end
end

score_str(x)::String = lpad(clamp(log_score(x), 0, 9999), 4, '0')
scaled_score_str(x)::String = lpad(clamp(scaled_log_score(x), 0, 9999), 4, '0')

rktk_score_str(opt::RKOCBFGSOptimizer{T}) where {T <: Real} =
    score_str(opt.objective[1]) * '-' * score_str(norm(opt.gradient)) *
        '-' * scaled_score_str(norm(opt.current_point))

rktk_filename(opt, id::RKTKID)::String =
    rktk_score_str(opt) * '-' * string(id) * ".txt"

numstr(x)::String = @sprintf("%#-18.12g", BigFloat(x))
shortstr(x)::String = @sprintf("%#.5g", BigFloat(x))

function print_help()::Nothing
    say("Usage: julia RKTK.jl <command> [parameters...]")
    say()
    say("RKTK provides the following <command> options:")
    say("    search <order> <num-stages> <precision>")
    say("    refine <rktk-id> <precision>")
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

function rkoc_optimizer(::Type{T}, order::Int, num_stages::Int,
        x_init::Vector{BigFloat}, num_iters::Int) where {T <: Real}
    obj_func, grad_func = rkoc_explicit_backprop_functors(T, order, num_stages)
    num_vars = div(num_stages * (num_stages + 1), 2)
    @assert length(x_init) == num_vars
    opt = BFGSOptimizer(T.(x_init), inv(T(1_000_000)), obj_func, grad_func)
    opt.iteration[1] = num_iters
    opt
end

function rkoc_optimizer(::Type{T}, id::RKTKID,
                        filename::String) where {T <: Real}
    trajectory = filter(!isempty, strip.(split(read(filename, String), "\n\n")))
    point_data = filter(!isempty, strip.(split(trajectory[end], '\n')))
    header = split(point_data[1])
    x_init = BigFloat.(point_data[2:end])
    num_iters = parse(Int, header[1])
    rkoc_optimizer(T, id.order, id.num_stages, x_init, num_iters), header
end

################################################################################

function save_to_file(opt::RKOCBFGSOptimizer{T}, id::RKTKID) where {T <: Real}
    filename = rktk_filename(opt, id)
    rmk("Saving progress to file ", filename, "...")
    old_filename = find_filename_by_id(".", id)
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

const TERM = isa(stdout, Base.TTY)

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
        elseif TERM && (current_time - last_print_time > UInt(80_000_000))
            rmk_table_row(opt, "BFGS")
            last_print_time = current_time
        end
    end
end

function run!(opt::RKOCBFGSOptimizer{T}, id::RKTKID,
              duration_ns::UInt) where {T <: Real}
    save_to_file(opt, id)
    start_time = last_save_time = time_ns()
    while true
        _, objective_decreased = step!(opt)
        current_time = time_ns()
        if (!objective_decreased) || (current_time - start_time > duration_ns)
            save_to_file(opt, id)
            return !objective_decreased
        end
        if current_time - last_save_time > UInt(60_000_000_000)
            save_to_file(opt, id)
            last_save_time = current_time
        end
    end
end

################################################################################

function search(::Type{T}, id::RKTKID) where {T <: Real}
    setprecision(approx_precision(T))
    num_vars = div(id.num_stages * (id.num_stages + 1), 2)
    optimizer = rkoc_optimizer(T, id.order, id.num_stages,
        rand(BigFloat, num_vars), 0)
    say("Running $T search $id.\n")
    run!(optimizer, id)
    say("\nCompleted $T search $id.\n")
end

# function multisearch(::Type{T}, order::Int, num_stages::Int) where {T <: Real}
#     num_threads = nthreads()
#     num_opts = num_threads - 1
#     num_vars = div(num_stages * (num_stages + 1), 2)
#     say("Constructing optimizers...")
#     optimizers = [rkoc_optimizer(
#             T, order, num_stages, rand(BigFloat, num_vars), 0)
#         for _ = 1 : num_opts]
#     bfgs_used = zeros(Bool, num_opts)
#     objective_decreased = zeros(Bool, num_opts)
#     for i = 1 : num_opts
#         print_table_row(optimizers[i], "NONE")
#         bfgs_used[i], objective_decreased[i] = step!(optimizers[i])
#     end
#     @threads for i = 1 : num_threads
#         if i <= num_opts
#             while true
#                 bfgs_used[i], objective_decreased[i] = step!(optimizers[i])
#             end
#         else
#             while true
#                 for i = 1 : num_opts
#                     print("\033[F")
#                 end
#                 for i = 1 : num_opts
#                     print_table_row(optimizers[i],
#                         ifelse(objective_decreased[i],
#                             ifelse(bfgs_used[i], "BFGS", "GRAD"), "DONE"))
#                 end

#             end
#         end
#     end
# end

function refine(::Type{T}, id::RKTKID, filename::String) where {T <: Real}
    setprecision(approx_precision(T))
    say("Running ", T, " refinement $id.\n")
    optimizer, header = rkoc_optimizer(T, id, filename)
    if precision(BigFloat) < parse(Int, header[2])
        say("WARNING: Refining at lower precision than source file.\n")
    end
    starting_iteration = optimizer.iteration[1]
    run!(optimizer, id)
    ending_iteration = optimizer.iteration[1]
    if ending_iteration > starting_iteration
        say("\nRepeating $T refinement $id.\n")
        refine(T, id, find_filename_by_id(".", id))
    else
        say("\nCompleted $T refinement $id.\n")
    end
end

function clean(::Type{T}) where {T <: Real}
    setprecision(approx_precision(T))
    optimizers = Tuple{Int,RKTKID,RKOCBFGSOptimizer{T}}[]
    for filename in readdir()
        if match(RKTKID_REGEX, filename) != nothing
            id = find_rktkid(filename)
            optimizer, header = rkoc_optimizer(T, id, filename)
            if precision(BigFloat) < parse(Int, header[2])
                say("ERROR: Cleaning at lower precision than source file \"",
                    filename, "\".")
                exit()
            end
            push!(optimizers,
                (log_score(optimizer.objective[1]), id, optimizer))
        end
    end
    say("Found ", length(optimizers), " RKTK files.")
    while true
        num_optimizers = length(optimizers)
        sort!(optimizers, by=(t -> t[1]), rev=true)
        completed = zeros(Bool, num_optimizers)
        @threads for i = 1 : num_optimizers
            _, id, optimizer = optimizers[i]
            old_score = rktk_score_str(optimizer)
            start_iter = optimizer.iteration[1]
            completed[i] = run!(optimizer, id, UInt(2_000_000_000))
            stop_iter = optimizer.iteration[1]
            new_score = rktk_score_str(optimizer)
            say(ifelse(completed[i], "    Cleaned ", "    Working "),
                id, " (", stop_iter - start_iter, " iterations: ",
                old_score, " => ", new_score, ") on thread ", threadid(), ".")
        end
        next_optimizers = Tuple{Int,RKTKID,RKOCBFGSOptimizer{T}}[]
        for i = 1 : length(optimizers)
            if !completed[i]
                _, id, optimizer = optimizers[i]
                push!(next_optimizers,
                    (log_score(optimizer.objective[1]), id, optimizer))
            end
        end
        if length(next_optimizers) == 0
            say("All RKTK files cleaned!")
            break
        end
        optimizers = next_optimizers
        say(length(optimizers), " RKTK files remaining.")
    end
end

function benchmark(::Type{T}, order::Int, num_stages::Int,
        benchmark_secs) where {T <: Real}
    setprecision(approx_precision(T))
    num_vars = div(num_stages * (num_stages + 1), 2)
    x_init = [BigFloat(i) / num_vars for i = 1 : num_vars]
    construction_ns = time_ns()
    opt = rkoc_optimizer(T, order, num_stages, x_init, 0)
    start_ns = time_ns()
    benchmark_ns = round(typeof(start_ns), benchmark_secs * 1_000_000_000)
    terminated_early = false
    while time_ns() - start_ns < benchmark_ns
        _, made_progress = step!(opt)
        if !made_progress
            terminated_early = true
            break
        end
    end
    Int(start_ns - construction_ns), opt.iteration[1], terminated_early
end

function benchmark(::Type{T}, order::Int, num_stages::Int,
        benchmark_secs, num_trials::Int) where {T <: Real}
    construction_secs = Float64[]
    iteration_counts = Int[]
    success = true
    for _ = 1 : num_trials
        construction_ns, iteration_count, terminated_early =
            benchmark(T, order, num_stages, benchmark_secs / num_trials)
        push!(construction_secs, construction_ns / 1_000_000_000)
        push!(iteration_counts, iteration_count)
        if terminated_early
            success = false
            break
        end
    end
    if success
        say(rpad("$T: ", 16, ' '),
            shortstr(mean(iteration_counts)), " ± ",
            shortstr(std(iteration_counts)))
    else
        say(rpad("$T: ", 16, ' '), "Search terminated too early")
    end
end

################################################################################

function get_order(n::Int)
    result = tryparse(Int, ARGS[n])
    if (result == nothing) || (result < 1) || (result > 20)
        say("ERROR: Parameter $n (\"$(ARGS[n])\") must be ",
            "an integer between 1 and 20.")
        exit()
    end
    result
end

function get_num_stages(n::Int)
    result = tryparse(Int, ARGS[n])
    if (result == nothing) || (result < 1) || (result > 99)
        say("ERROR: Stage parameter $n (\"$(ARGS[n])\") must be ",
            "an integer between 1 and 99.")
        exit()
    end
    result
end

function main()

    if (length(ARGS) == 0) || ("-h" in ARGS) || ("--help" in ARGS) let
        print_help()

    end elseif uppercase(ARGS[1]) == "SEARCH" let
        order, num_stages = get_order(2), get_num_stages(3)
        prec = parse(Int, ARGS[4])
        while true
            search(precision_type(prec), RKTKID(order, num_stages, uuid4()))
        end

    # end elseif uppercase(ARGS[1]) == "MULTISEARCH" let
    #     order, num_stages = get_order(2), get_num_stages(3)
    #     prec = parse(Int, ARGS[4])
    #     multisearch(precision_type(prec), order, num_stages)

    end elseif uppercase(ARGS[1]) == "REFINE" let
        id = find_rktkid(ARGS[2])
        if id == nothing
            say("ERROR: Invalid RKTK ID ", ARGS[2], ".")
            say("RKTK IDs have the form ",
                "RKTK-XXYY-ZZZZZZZZ-ZZZZ-ZZZZ-ZZZZ-ZZZZZZZZZZZZ.\n")
            exit()
        end
        filename = find_filename_by_id(".", id)
        if filename == nothing
            say("ERROR: No file exists with RKTK ID $id.\n")
            exit()
        end
        prec = parse(Int, ARGS[3])
        refine(precision_type(prec), id, filename)

    end elseif uppercase(ARGS[1]) == "CLEAN" let
        prec = parse(Int, ARGS[2])
        clean(precision_type(prec))

    end elseif uppercase(ARGS[1]) == "BENCHMARK" let
        order, num_stages = get_order(2), get_num_stages(3)
        benchmark_secs = parse(Float64, ARGS[4])
        num_trials = parse(Int, ARGS[5])
        for T in (Float32, Float64, Float64x2, Float64x3, Float64x4,
                    Float64x5, Float64x6, Float64x7, Float64x8)
            benchmark(T, order, num_stages, benchmark_secs, num_trials)
        end
        for T in (Float32, Float64, Float64x2, Float64x3, Float64x4,
                    Float64x5, Float64x6, Float64x7, Float64x8)
            setprecision(approx_precision(T))
            benchmark(BigFloat, order, num_stages, benchmark_secs, num_trials)
        end
        say()

    end else let
        say("ERROR: Unrecognized <command> option \"", ARGS[1], "\".\n")
        print_help()

    end end
end

main()

# const AccurateReal = Float64x4
# const THRESHOLD = AccurateReal(1e-40)

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
