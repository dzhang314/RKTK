using Base.Threads
using DZOptimization
using DZOptimization.Kernels: norm2
using MultiFloats
using Printf
using RungeKuttaToolKit
using Serialization


const OptimizerAE{T} = LBFGSOptimizer{typeof(DZOptimization.NULL_CONSTRAINT),
    RKOCEvaluatorAE{T},RungeKuttaToolKit.RKOCEvaluatorAEAdjoint{T},QuadraticLineSearch,T,1}
const OptimizerAI{T} = LBFGSOptimizer{typeof(DZOptimization.NULL_CONSTRAINT),
    RKOCEvaluatorAI{T},RungeKuttaToolKit.RKOCEvaluatorAIAdjoint{T},QuadraticLineSearch,T,1}
const OptimizerBE{T} = LBFGSOptimizer{typeof(DZOptimization.NULL_CONSTRAINT),
    RKOCEvaluatorBE{T},RungeKuttaToolKit.RKOCEvaluatorBEAdjoint{T},QuadraticLineSearch,T,1}
const OptimizerBI{T} = LBFGSOptimizer{typeof(DZOptimization.NULL_CONSTRAINT),
    RKOCEvaluatorBI{T},RungeKuttaToolKit.RKOCEvaluatorBIAdjoint{T},QuadraticLineSearch,T,1}


@assert length(ARGS) == 4
const MODE = ARGS[1]
@assert length(MODE) == 4
const RESIDUAL_SCORE_THRESHOLD = parse(Int, ARGS[2])
@assert 0 <= RESIDUAL_SCORE_THRESHOLD <= 9999
const GRADIENT_SCORE_THRESHOLD = parse(Int, ARGS[3])
@assert 0 <= GRADIENT_SCORE_THRESHOLD <= 9999
const NORM_SCORE_THRESHOLD = parse(Int, ARGS[4])
@assert 0 <= NORM_SCORE_THRESHOLD <= 9999


@static if MODE[3:4] == "M1"
    const FloatType = Float64x2
    const NEXT_MODE = MODE[1:2] * "M2"
elseif MODE[3:4] == "M2"
    const FloatType = Float64x3
    const NEXT_MODE = MODE[1:2] * "M3"
elseif MODE[3:4] == "M3"
    const FloatType = Float64x4
    const NEXT_MODE = MODE[1:2] * "M4"
elseif MODE[3:4] == "M4"
    const FloatType = Float64x5
    const NEXT_MODE = MODE[1:2] * "M5"
elseif MODE[3:4] == "M5"
    const FloatType = Float64x6
    const NEXT_MODE = MODE[1:2] * "M6"
elseif MODE[3:4] == "M6"
    const FloatType = Float64x7
    const NEXT_MODE = MODE[1:2] * "M7"
elseif MODE[3:4] == "M7"
    const FloatType = Float64x8
    const NEXT_MODE = MODE[1:2] * "M8"
elseif MODE[3:4] == "M8"
    const FloatType = BigFloat
    setprecision(BigFloat, 512)
    const NEXT_MODE = MODE[1:2] * "X1"
elseif MODE[3:4] == "X1"
    const FloatType = BigFloat
    setprecision(BigFloat, 1024)
    const NEXT_MODE = MODE[1:2] * "X2"
elseif MODE[3:4] == "X2"
    const FloatType = BigFloat
    setprecision(BigFloat, 2048)
    const NEXT_MODE = MODE[1:2] * "X3"
elseif MODE[3:4] == "X3"
    const FloatType = BigFloat
    setprecision(BigFloat, 4096)
    const NEXT_MODE = MODE[1:2] * "X4"
elseif MODE[3:4] == "X4"
    const FloatType = BigFloat
    setprecision(BigFloat, 8192)
    const NEXT_MODE = MODE[1:2] * "X5"
elseif MODE[3:4] == "X5"
    const FloatType = BigFloat
    setprecision(BigFloat, 16384)
    const NEXT_MODE = MODE[1:2] * "X6"
elseif MODE[3:4] == "X6"
    const FloatType = BigFloat
    setprecision(BigFloat, 32768)
    const NEXT_MODE = MODE[1:2] * "X7"
elseif MODE[3:4] == "X7"
    const FloatType = BigFloat
    setprecision(BigFloat, 65536)
    const NEXT_MODE = MODE[1:2] * "X8"
elseif MODE[3:4] == "X8"
    const FloatType = BigFloat
    setprecision(BigFloat, 131072)
    const NEXT_MODE = MODE[1:2] * "X9"
else
    error("Unrecognized mode $MODE.")
end


@static if MODE[1:2] == "AE"
    const OptimizerType = OptimizerAE{FloatType}
    const EvaluatorType = RKOCEvaluatorAE{FloatType}
elseif MODE[1:2] == "AI"
    const OptimizerType = OptimizerAI{FloatType}
    const EvaluatorType = RKOCEvaluatorAI{FloatType}
elseif MODE[1:2] == "BE"
    const OptimizerType = OptimizerBE{FloatType}
    const EvaluatorType = RKOCEvaluatorBE{FloatType}
elseif MODE[1:2] == "BI"
    const OptimizerType = OptimizerBI{FloatType}
    const EvaluatorType = RKOCEvaluatorBI{FloatType}
end


const RKTK_TXT_FILENAME_REGEX =
    r"^RKTK-([0-9]{2})-([0-9]{2})-([AB][EI][MX][0-9])-([0-9]{4})-([0-9]{4})-([0-9]{4}|FAIL)-([0-9A-Fa-f]{16})\.txt$"
const RKTK_JLS_FILENAME_REGEX =
    r"^RKTK-([0-9]{2})-([0-9]{2})-([AB][EI][MX][0-9])-([0-9]{4})-([0-9]{4})-([0-9]{4})-([0-9]{12})-([0-9A-Fa-f]{16})\.jls$"


const RKTKRecord = Tuple{OptimizerType,UInt64,Vector{Pair{String,Int}}}


function compute_scores(optimizer::OptimizerType)
    num_residuals = length(optimizer.objective_function.residuals)
    num_variables = length(optimizer.current_point)
    rms_residual = sqrt(optimizer.current_objective_value[] / num_residuals)
    rms_gradient = sqrt(norm2(optimizer.current_gradient) / num_variables)
    rms_coeff = sqrt(norm2(optimizer.current_point) / num_variables)
    residual_score = round(Int,
        clamp(-500 * log10(Float64(rms_residual)), 0.0, 9999.0))
    gradient_score = round(Int,
        clamp(-500 * log10(Float64(rms_gradient)), 0.0, 9999.0))
    coeff_score = round(Int,
        clamp(10000 - 2500 * log10(Float64(rms_coeff)), 0.0, 9999.0))
    return (residual_score, gradient_score, coeff_score)
end


function compute_jls_filename(order::Int, num_stages::Int, record::RKTKRecord)
    optimizer, seed, iteration_counts = record
    residual_score, gradient_score, norm_score = compute_scores(optimizer)
    return @sprintf("RKTK-%02d-%02d-%s-%04d-%04d-%04d-%s-%016X.jls",
        order, num_stages, NEXT_MODE,
        residual_score, gradient_score, norm_score,
        lpad(sum(n for (_, n) in iteration_counts), 12,
            optimizer.has_terminated[] ? '0' : 'X'), seed)
end


function main()

    orders = BitSet()
    stages = BitSet()
    available_txt_files = Dict{UInt64,String}()
    available_jls_files = Dict{UInt64,String}()
    available_next_jls_files = Dict{UInt64,String}()

    for filename in readdir()

        m = match(RKTK_TXT_FILENAME_REGEX, filename)
        if !isnothing(m)

            order = parse(Int, m[1]; base=10)
            push!(orders, order)
            @assert isone(length(orders))

            num_stages = parse(Int, m[2]; base=10)
            push!(stages, num_stages)
            @assert isone(length(stages))

            mode = m[3]
            residual_score = parse(Int, m[4]; base=10)
            gradient_score = parse(Int, m[5]; base=10)
            norm_score = (m[6] == "FAIL") ? nothing : parse(Int, m[6])
            seed = parse(UInt64, m[7]; base=16)

            if ((mode == MODE) &&
                (residual_score >= RESIDUAL_SCORE_THRESHOLD) &&
                (gradient_score >= GRADIENT_SCORE_THRESHOLD) &&
                (!isnothing(norm_score)) &&
                (norm_score >= NORM_SCORE_THRESHOLD))
                @assert !haskey(available_txt_files, seed)
                available_txt_files[seed] = filename
            end
        end

        m = match(RKTK_JLS_FILENAME_REGEX, filename)
        if !isnothing(m)

            order = parse(Int, m[1]; base=10)
            push!(orders, order)
            @assert isone(length(orders))

            num_stages = parse(Int, m[2]; base=10)
            push!(stages, num_stages)
            @assert isone(length(stages))

            mode = m[3]
            residual_score = parse(Int, m[4]; base=10)
            gradient_score = parse(Int, m[5]; base=10)
            norm_score = parse(Int, m[6]; base=10)
            total_iteration_count = parse(Int, m[7]; base=10)
            seed = parse(UInt64, m[8]; base=16)

            if ((mode == MODE) &&
                (residual_score >= RESIDUAL_SCORE_THRESHOLD) &&
                (gradient_score >= GRADIENT_SCORE_THRESHOLD) &&
                (norm_score >= NORM_SCORE_THRESHOLD))
                @assert !haskey(available_jls_files, seed)
                available_jls_files[seed] = filename
            end

            if ((mode == NEXT_MODE) &&
                (residual_score >= RESIDUAL_SCORE_THRESHOLD) &&
                (gradient_score >= GRADIENT_SCORE_THRESHOLD) &&
                (norm_score >= NORM_SCORE_THRESHOLD))
                @assert !haskey(available_next_jls_files, seed)
                available_next_jls_files[seed] = filename
            end
        end
    end

    if (isempty(available_txt_files) &&
        isempty(available_jls_files) &&
        isempty(available_next_jls_files))
        @printf("No eligible RKTK files found.\n")
        return nothing
    end

    order = first(orders)
    num_stages = first(stages)
    @printf("Performing RKTK-%02d-%02d-%s refinement.\n",
        order, num_stages, NEXT_MODE)
    @printf("Using %04d-%04d-%04d score threshold.\n",
        RESIDUAL_SCORE_THRESHOLD, GRADIENT_SCORE_THRESHOLD,
        NORM_SCORE_THRESHOLD)
    @printf("Found %d eligible .txt files.\n", length(available_txt_files))
    @printf("Found %d eligible .jls files.\n", length(available_jls_files))

    records = Vector{RKTKRecord}()

    for (seed, filename) in available_txt_files
        if (!haskey(available_jls_files, seed) &&
            !haskey(available_next_jls_files, seed))

            # Text files should only be produced at machine precision.
            @assert MODE[3:4] == "M1"

            parts = split(read(filename, String), "\n\n")
            @assert length(parts) == 3
            initial_part, table, final_part = parts

            initial_lines = split(initial_part, '\n')
            final_lines = split(final_part, '\n')
            @assert length(initial_lines) + 1 == length(final_lines)
            @assert isempty(final_lines[end])
            initial_point = parse.(Float64, initial_lines)
            final_point = parse.(Float64, final_lines[1:end-1])

            table_entries = strip.(split(split(table, '\n')[end], '|'))
            @assert length(table_entries) == 7
            @assert isempty(table_entries[1])
            iteration_count = parse(Int, table_entries[2])

            evaluator = EvaluatorType(order, num_stages)
            optimizer = LBFGSOptimizer(evaluator, evaluator',
                QuadraticLineSearch(), FloatType.(final_point),
                sqrt(eps(FloatType) * length(final_point)),
                length(final_point))

            push!(records, (optimizer, seed,
                [MODE => iteration_count, NEXT_MODE => 0]))
        end
    end

    for (seed, filename) in available_jls_files
        if !haskey(available_next_jls_files, seed)
            record = deserialize(filename)
            @assert length(record) == 3
            final_point = record[1].current_point
            @assert record[2] == seed
            iteration_counts = record[3]
            evaluator = EvaluatorType(order, num_stages)
            optimizer = LBFGSOptimizer(evaluator, evaluator',
                QuadraticLineSearch(), FloatType.(final_point),
                sqrt(eps(FloatType) * length(final_point)),
                length(final_point))
            push!(iteration_counts, NEXT_MODE => 0)
            push!(records, (optimizer, seed, iteration_counts))
        end
    end

    for (_, filename) in available_next_jls_files
        push!(records, deserialize(filename))
    end

    @printf("Loaded %d records.\n", length(records))

    for record in records
        _, seed, _ = record
        if !haskey(available_next_jls_files, seed)
            filename = compute_jls_filename(order, num_stages, record)
            @assert !isfile(filename)
            serialize(filename, record)
            available_next_jls_files[seed] = filename
        end
    end

    while !isempty(records)
        @threads for record in records
            optimizer, seed, iteration_counts = record

            old_scores = compute_scores(optimizer)
            start_iteration = optimizer.iteration_count[]
            start_time = time_ns()
            for _ = 1:100
                if optimizer.has_terminated[]
                    break
                end
                step!(optimizer)
            end
            end_time = time_ns()
            end_iteration = optimizer.iteration_count[]
            new_scores = compute_scores(optimizer)

            @assert end_iteration >= start_iteration
            if end_iteration == start_iteration
                @assert optimizer.has_terminated[]
            end

            elapsed_time = (end_time - start_time) / 1.0e9
            @printf("Refined seed %016X from %04d-%04d-%04d to %04d-%04d-%04d.\nPerformed %d iterations in %g seconds (%g iterations per second).\n",
                seed, old_scores..., new_scores..., end_iteration - start_iteration, elapsed_time, (end_iteration - start_iteration) / elapsed_time)

            mode, _ = iteration_counts[end]
            @assert mode == NEXT_MODE
            iteration_counts[end] = mode => optimizer.iteration_count[]
            filename = compute_jls_filename(order, num_stages, record)
            @assert !isfile(filename)
            serialize(filename, record)
            rm(available_next_jls_files[seed])
            available_next_jls_files[seed] = filename
            @printf("Wrote %s to disk.\n", filename)
        end
        filter!(((optimizer, _, _),) -> !optimizer.has_terminated[], records)
    end
end


main()
