println(raw"                ______ _   _______ _   __")
println(raw"                | ___ \ | / /_   _| | / /   Version  2.3")
println(raw"                | |_/ / |/ /  | | | |/ /")
println(raw"                |    /|    \  | | |    \   David K. Zhang")
println(raw"                | |\ \| |\  \ | | | |\  \     (c) 2022")
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

using DZOptimization
using DZOptimization: norm
using MultiFloats
using RungeKuttaToolKit

push!(LOAD_PATH, @__DIR__)
using DZMisc

set_zero_subnormals(true)
use_standard_multifloat_arithmetic()

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
    if m !== nothing
        RKTKID(parse(Int, m[1]), parse(Int, m[2]), UUID(m[3]))
    else
        nothing
    end
end

function find_filename_by_id(dir::String, id::RKTKID)::Union{String,Nothing}
    result = nothing
    for filename in readdir(dir)
        m = match(RKTKID_REGEX, filename)
        if ((m !== nothing) && (parse(Int, m[1]) == id.order) &&
                (parse(Int, m[2]) == id.num_stages) && (UUID(m[3]) == id.uuid))
            if result !== nothing
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

rktk_score_str(opt) = *(
    score_str(opt.current_objective_value[]),
    '-',
    score_str(norm(opt.current_gradient)),
    '-',
    scaled_score_str(norm(opt.current_point))
)

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
    print_table_row(opt.iteration_count[], opt.current_objective_value[], norm(opt.current_gradient),
                    opt.last_step_size[1], norm(opt.current_point), type)
end

function rmk_table_row(opt, type)::Nothing
    rmk_table_row(opt.iteration_count[], opt.current_objective_value[], norm(opt.current_gradient),
                  opt.last_step_size[1], norm(opt.current_point), type)
end

################################################################################

function rkoc_optimizer(::Type{T}, order::Int, num_stages::Int,
        x_init::Vector{BigFloat}, num_iters::Int) where {T <: Real}
    obj_func, grad_func = rkoc_explicit_backprop_functors(T, order, num_stages)
    num_vars = div(num_stages * (num_stages + 1), 2)
    @assert length(x_init) == num_vars
    opt = BFGSOptimizer(obj_func, grad_func, T.(x_init), inv(T(1_000_000)))
    opt.iteration_count[] = num_iters
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

function save_to_file(opt, id::RKTKID) where {T <: Real}
    filename = rktk_filename(opt, id)
    rmk("Saving progress to file ", filename, "...")
    old_filename = find_filename_by_id(".", id)
    if old_filename !== nothing
        if old_filename != filename
            mv(old_filename, filename)
        end
        trajectory = filter(!isempty,
            strip.(split(read(filename, String), "\n\n")))
        point_data = filter(!isempty, strip.(split(trajectory[end], '\n')))
        header = split(point_data[1])
        num_iters = parse(Int, header[1])
        prec = parse(Int, header[2])
        if (precision(BigFloat) <= prec) && (opt.iteration_count[] <= num_iters)
            rmk("No save necessary.")
            return
        end
    end
    file = open(filename, "a+")
    println(file, opt.iteration_count[], ' ', precision(BigFloat), ' ',
            log_score(opt.current_objective_value[]), ' ', log_score(norm(opt.current_gradient)))
    for x in opt.current_point
        println(file, BigFloat(x))
    end
    println(file)
    close(file)
    rmk("Save complete.")
end

const TERM = isa(stdout, Base.TTY)

function run!(opt, id::RKTKID) where {T <: Real}
    print_table_header()
    print_table_row(opt, "NONE")
    save_to_file(opt, id)
    last_print_time = last_save_time = time_ns()
    while true
        step!(opt)
        if opt.has_converged[]
            print_table_row(opt, "DONE")
            save_to_file(opt, id)
            return
        end
        current_time = time_ns()
        if current_time - last_save_time > UInt(60_000_000_000)
            save_to_file(opt, id)
            last_save_time = current_time
        end
        if opt.last_step_type[] == DZOptimization.GradientDescentStep
            print_table_row(opt, "GRAD")
            last_print_time = current_time
        elseif TERM && (current_time - last_print_time > UInt(80_000_000))
            rmk_table_row(opt, "BFGS")
            last_print_time = current_time
        end
    end
end

function run!(opt, id::RKTKID, duration_ns::UInt) where {T <: Real}
    save_to_file(opt, id)
    start_time = last_save_time = time_ns()
    while true
        step!(opt)
        current_time = time_ns()
        if opt.has_converged[] || (current_time - start_time > duration_ns)
            save_to_file(opt, id)
            return opt.has_converged[]
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

function refine(::Type{T}, id::RKTKID, filename::String) where {T <: Real}
    setprecision(approx_precision(T))
    say("Running ", T, " refinement $id.\n")
    optimizer, header = rkoc_optimizer(T, id, filename)
    if precision(BigFloat) < parse(Int, header[2])
        say("WARNING: Refining at lower precision than source file.\n")
    end
    starting_iteration = optimizer.iteration_count[]
    run!(optimizer, id)
    ending_iteration = optimizer.iteration_count[]
    if ending_iteration > starting_iteration
        say("\nRepeating $T refinement $id.\n")
        refine(T, id, find_filename_by_id(".", id))
    else
        say("\nCompleted $T refinement $id.\n")
    end
end

function clean(::Type{T}) where {T <: Real}
    setprecision(approx_precision(T))
    optimizers = Tuple{Int,RKTKID,Any}[]
    for filename in readdir()
        if match(RKTKID_REGEX, filename) !== nothing
            id = find_rktkid(filename)
            optimizer, header = rkoc_optimizer(T, id, filename)
            if precision(BigFloat) < parse(Int, header[2])
                say("ERROR: Cleaning at lower precision than source file \"",
                    filename, "\".")
                exit()
            end
            push!(optimizers,
                (log_score(optimizer.current_objective_value[]), id, optimizer))
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
            start_iter = optimizer.iteration_count[]
            completed[i] = run!(optimizer, id, UInt(10_000_000_000))
            stop_iter = optimizer.iteration_count[]
            new_score = rktk_score_str(optimizer)
            say(ifelse(completed[i], "    Cleaned ", "    Working "),
                id, " (", stop_iter - start_iter, " iterations: ",
                old_score, " => ", new_score, ") on thread ", threadid(), ".")
        end
        next_optimizers = Tuple{Int,RKTKID,Any}[]
        for i = 1 : length(optimizers)
            if !completed[i]
                _, id, optimizer = optimizers[i]
                push!(next_optimizers,
                    (log_score(optimizer.current_objective_value[]), id, optimizer))
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
        step!(opt)
        if opt.has_converged[]
            terminated_early = true
            break
        end
    end
    Int(start_ns - construction_ns), opt.iteration_count[], terminated_early
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
    if (result === nothing) || (result < 1) || (result > 20)
        say("ERROR: Parameter $n (\"$(ARGS[n])\") must be ",
            "an integer between 1 and 20.")
        exit()
    end
    result
end

function get_num_stages(n::Int)
    result = tryparse(Int, ARGS[n])
    if (result === nothing) || (result < 1) || (result > 99)
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
        if id === nothing
            say("ERROR: Invalid RKTK ID ", ARGS[2], ".")
            say("RKTK IDs have the form ",
                "RKTK-XXYY-ZZZZZZZZ-ZZZZ-ZZZZ-ZZZZ-ZZZZZZZZZZZZ.\n")
            exit()
        end
        filename = find_filename_by_id(".", id)
        if filename === nothing
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
