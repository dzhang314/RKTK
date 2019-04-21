module DZMisc

export rmk, say, dbl, scale, integer_partitions,
    RootedTree, rooted_trees, rooted_tree_count, butcher_density,
    orthonormalize_columns!, linearly_independent_column_indices!,
    norm, norm2, normalize!, approx_norm, approx_norm2, approx_normalize!,
    dot, identity_matrix!,
    quadratic_line_search, quadratic_search,
    asm_lines, asm_calls, view_asm

using InteractiveUtils: _dump_function

################################################################################

function rmk(args...; verbose::Bool=true)::Nothing
    if verbose
        print("\33[2K\r")
        print(args...)
        flush(stdout)
    end
end

function say(args...; verbose::Bool=true)::Nothing
    if verbose
        print("\33[2K\r")
        println(args...)
        flush(stdout)
    end
end

################################################################################

@inline function dbl(x::T)::T where {T <: Number}
    x + x
end

@inline function scale(a::Float64, x::T)::T where {T <: Number}
    a * x
end

################################################################################

function _int_parts_impl(n::Int, max::Int)::Vector{Vector{Pair{Int,Int}}}
    if n == 0
        [Pair{Int,Int}[]]
    elseif max == 0
        Vector{Pair{Int,Int}}[]
    else
        result = Vector{Pair{Int,Int}}[]
        for m = max : -1 : 1, q = div(n, m) : -1 : 1
            for p in _int_parts_impl(n - q * m, m - 1)
                push!(result, push!(p, m => q))
            end
        end
        result
    end
end

function _int_parts_impl(n::Int, max::Int, len::Int)::Vector{Vector{Int}}
    if n == 0
        [zeros(Int, len)]
    elseif max * len < n
        Vector{Int}[]
    else
        result = Vector{Int}[]
        for m = min(n, max) : -1 : div(n + len - 1, len)
            for p in _int_parts_impl(n - m, m, len - 1)
                push!(result, push!(p, m))
            end
        end
        result
    end
end

integer_partitions(n::Int)::Vector{Vector{Pair{Int,Int}}} =
    reverse!.(_int_parts_impl(n, n))
integer_partitions(n::Int, len::Int)::Vector{Vector{Int}} =
    reverse!.(_int_parts_impl(n, n, len))

################################################################################

function next_permutation!(items::Vector{T})::Bool where {T}
    num_items = length(items)
    if num_items == 0; return false; end
    current_item = items[num_items]
    pivot_index = num_items - 1
    while pivot_index != 0
        next_item = items[pivot_index]
        if next_item >= current_item
            pivot_index -= 1
            current_item = next_item
        else; break; end
    end
    if pivot_index == 0; return false; end
    pivot = items[pivot_index]
    successor_index = num_items
    while items[successor_index] <= pivot; successor_index -= 1; end
    items[pivot_index], items[successor_index] =
        items[successor_index], items[pivot_index]
    reverse!(view(items, pivot_index + 1 : num_items))
    return true
end

function previous_permutation!(items::Vector{T})::Bool where {T}
    num_items = length(items)
    if num_items == 0; return false; end
    current_item = items[num_items]
    pivot_index = num_items - 1
    while pivot_index != 0
        next_item = items[pivot_index]
        if next_item <= current_item
            pivot_index -= 1
            current_item = next_item
        else; break; end
    end
    if pivot_index == 0; return false; end
    pivot = items[pivot_index]
    successor_index = num_items
    while items[successor_index] >= pivot; successor_index -= 1; end
    items[pivot_index], items[successor_index] =
        items[successor_index], items[pivot_index]
    reverse!(view(items, pivot_index + 1 : num_items))
    return true
end

function combinations_with_replacement(
        items::Vector{T}, n::Int)::Vector{Vector{Pair{T,Int}}} where {T}
        combinations = Vector{Pair{T,Int}}[]
    for p in integer_partitions(n, length(items))
        while true
            comb = Pair{T,Int}[]
            for (item, k) in zip(items, p)
                if k > 0; push!(comb, item => k); end
            end
            push!(combinations, comb)
            if !previous_permutation!(p); break; end
        end
    end
    combinations
end

################################################################################

struct RootedTree
    order::Int
    children::Vector{Pair{RootedTree,Int}}
end

function rooted_trees(n::Int)::Vector{Vector{RootedTree}}
    result = Vector{RootedTree}[]
    for k = 1 : n
        trees = RootedTree[]
        for partition in integer_partitions(k - 1)
            combination_candidates = [
                combinations_with_replacement(result[order], multiplicity)
                for (order, multiplicity) in partition]
            for combination_sequence in Base.product(combination_candidates...)
                push!(trees, RootedTree(k, vcat(combination_sequence...)))
            end
        end
        push!(result, trees)
    end
    result
end

function rooted_tree_count(n::Int)::BigInt
    if n < 0; error("rooted_tree_count requires non-negative argument"); end
    if n == 0; return 0; end
    counts = zeros(Rational{BigInt}, n)
    counts[1] = 1
    for i = 2 : n, k = 1 : i - 1, m = 1 : div(i - 1, k)
        @inbounds counts[i] += k * counts[k] * counts[i - k * m] // (i - 1)
    end
    @assert all(isone(denominator(c)) for c in counts)
    sum(numerator.(counts))
end

function butcher_density(tree::RootedTree)::Int
    result = tree.order
    for (subtree, multiplicity) in tree.children
        result *= butcher_density(subtree)^multiplicity
    end
    result
end

function butcher_symmetry(tree::RootedTree)::Int
    result = 1
    for (subtree, multiplicity) in tree.children
        result *= butcher_symmetry(subtree) * factorial(multiplicity)
    end
    result
end

################################################################################

function orthonormalize_columns!(mat::Matrix{T})::Nothing where {T <: Real}
    m, n = size(mat, 1), size(mat, 2)
    for j = 1 : n
        let # normalize j'th column
            acc = zero(T)
            @simd for i = 1 : m
                @inbounds acc += abs2(mat[i, j])
            end
            acc = inv(sqrt(acc))
            @simd ivdep for i = 1 : m
                @inbounds mat[i, j] *= acc
            end
        end
        for k = j + 1 : n # orthogonalize k'th column against j'th column
            acc = zero(T)
            @simd for i = 1 : m
                @inbounds acc += mat[i, j] * mat[i, k]
            end
            @simd ivdep for i = 1 : m
                @inbounds mat[i, k] -= acc * mat[i, j]
            end
        end
    end
end

function linearly_independent_column_indices!(
        mat::Matrix{T}, threshold::T) where {T <: Real}
    indices = Int[]
    lo, hi, m, n = zero(T), T(Inf), size(mat, 1), size(mat, 2)
    for i = 1 : n
        x = zero(T)
        @simd for k = 1 : m
            @inbounds x += abs2(mat[k, i])
        end
        x = sqrt(x)
        if x > threshold
            hi = min(hi, x)
            push!(indices, i)
            y = inv(x)
            @simd ivdep for k = 1 : m
                @inbounds mat[k, i] *= y
            end
            for j = i + 1 : n
                acc = zero(T)
                @simd for k = 1 : m
                    @inbounds acc += mat[k, i] * mat[k, j]
                end
                @simd ivdep for k = 1 : m
                    @inbounds mat[k, j] -= acc * mat[k, i]
                end
            end
        else
            lo = max(lo, x)
        end
    end
    indices, lo, hi
end

################################################################################

@inline function norm2(x::Vector{T})::T where {T <: Number}
    result = zero(float(real(T)))
    @simd for i = 1 : length(x)
        @inbounds result += abs2(x[i])
    end
    result
end

@inline function norm(x::Vector{T})::T where {T <: Number}
    sqrt(norm2(x))
end

@inline function normalize!(x::Vector{T})::Nothing where {T <: Number}
    a = inv(norm(x))
    @simd ivdep for i = 1 : length(x)
        @inbounds x[i] *= a
    end
end

################################################################################

@inline function approx_norm2(x::Vector{T})::Float64 where {T <: Number}
    result = zero(Float64)
    @simd for i = 1 : length(x)
        @inbounds result += abs2(Float64(x[i]))
    end
    result
end

@inline function approx_norm(x::Vector{T})::Float64 where {T <: Number}
    sqrt(approx_norm2(x))
end

@inline function approx_normalize!(x::Vector{T})::Nothing where {T <: Number}
    a = inv(approx_norm(x))
    @simd ivdep for i = 1 : length(x)
        @inbounds x[i] *= a
    end
end

################################################################################

@inline function dot(v::Vector{T}, w::Vector{T}) where {T <: Real}
    result = zero(T)
    @simd ivdep for i = 1 : length(v)
        @inbounds result += v[i] * w[i]
    end
    result
end

@inline function identity_matrix!(A::Matrix{T}) where {T <: Number}
    m, n = size(A, 1), size(A, 2)
    for j = 1 : n
        @simd ivdep for i = 1 : m
            @inbounds A[i,j] = T(i == j)
        end
    end
end

################################################################################

function _qls_best(fb, x1, f1, x2, f2, x3, f3)
    xb = zero(x1)
    if f1 < fb
        xb, fb = x1, f1
    end
    if f2 < fb
        xb, fb = x2, f2
    end
    if f3 < fb
        xb, fb = x3, f3
    end
    xb, fb
end

function _qls_minimum_high(f0, f1, f2)
    q1 = f1 + f1
    q2 = q1 + q1
    q3 = f0 + f0
    q4 = f2 + f0
    q5 = q1 - q4
    (q2 - q3 - q4) / (q5 + q5)
end

function _qls_minimum_low(f0, f1, f2)
    q1 = f2 + f2
    q2 = q1 + q1
    q3 = f0 + f0
    q4 = f0 + f1
    q5 = q4 - q1
    q6 = q5 + q5
    (q4 + q3 - q2) / (q6 + q6)
end

function quadratic_line_search(f, f0, x1, args...)
    if isnan(f0)
        return zero(x1), f0
    end
    f1 = f(x1, args...)
    if isnan(f1)
        return quadratic_line_search(f, f0, scale(0.5, x1), args...)
    end
    if f1 < f0
        while true
            x2 = scale(2.0, x1)
            f2 = f(x2, args...)
            if (f2 >= f1) || isnan(f2)
                x3 = x1 * _qls_minimum_high(f0, f1, f2)
                f3 = f(x3, args...)
                return _qls_best(f0, x1, f1, x2, f2, x3, f3)
            else
                x1, f1 = x2, f2
            end
        end
    else
        while true
            x2 = scale(0.5, x1)
            f2 = f(x2, args...)
            if isnan(f2)
                return zero(x1), f0
            end
            if f2 <= f0
                x3 = x1 * _qls_minimum_low(f0, f1, f2)
                f3 = f(x3, args...)
                return _qls_best(f0, x2, f2, x1, f1, x3, f3)
            else
                x1, f1 = x2, f2
            end
        end
    end
end

function quadratic_search(f, x1, args...)
    quadratic_line_search(f, f(zero(x1), args...), x1, args...)
end

################################################################################

function asm_lines(@nospecialize(func), @nospecialize(types))
    # Note: _dump_function is an undocumented internal function that might
    # be changed or removed in future versions of Julia. This function is
    # only intended for interactive educational use.
    lines = split.(split(_dump_function(func, types,
        true,   # Generate native code (as opposed to LLVM IR).
        false,  # Don't generate wrapper code.
        true,   # This parameter (strip_ir_metadata) is ignored when dumping native code.
        true,   # This parameter (dump_module) is ignored when dumping native code.
        :intel, # I prefer Intel assembly syntax.
        true,   # This parameter (optimize) is ignored when dumping native code.
        :source # TODO: What does debuginfo=:source mean?
    ), '\n'))
    # I've dug through the Julia source code to try to determine the meaning
    # of the final parameter (debuginfo) of _dump_function, but it's hidden
    # behind more layers than I'd like to look (passed down into native code).
    # In the simple cases I've tested, it doesn't seem to make a difference.

    # Strip all empty lines and comments.
    filter!(line -> length(line) > 0 && !startswith(line[1], ';'), lines)
    return lines
end


const IGNORED_CALLS = ["jl_system_image_data", "jl_throw",
    "throw_complex_domainerror", "throw_overflowerr_binaryop"]
const IGNORED_PREFIXES = ["julia_throw_complex_domainerror_"]

function asm_calls(@nospecialize(func), @nospecialize(types))
    calls = Set{String}()
    for line in asm_lines(func, types) if "offset" in line
            call = line[end]
            ignored = (call in IGNORED_CALLS) || any(
                startswith(call, prefix) for prefix in IGNORED_PREFIXES)
            if !ignored; push!(calls, call); end
    end end
    calls
end

const MOV_TYPES = ["mov", "movabs", "vmovaps", "vmovups", "vmovapd", "vmovupd"]
const JUMP_TYPES = ["je", "jne", "ja", "jae", "jb", "jbe",
                    "jg", "jge", "jl", "jle"]
const JUMP_PREFIXES = Dict("je" => "", "jne" => "",
                           "ja" => "(unsigned) ", "jae" => "(unsigned) ",
                           "jb" => "(unsigned) ", "jbe" => "(unsigned) ",
                           "jg" => "(signed) ", "jge" => "(signed) ",
                           "jl" => "(signed) ", "jle" => "(signed) ")
const JUMP_SYMBOLS = Dict("je" => " == ", "jne" => " != ",
                          "ja" => " > ", "jae" => " >= ",
                          "jb" => " < ", "jbe" => " <= ",
                          "jg" => " > ", "jge" => " >= ",
                          "jl" => " < ", "jle" => " <= ")
const SHUFFLE_INSTRUCTIONS = ["vunpcklpd", "vunpckhpd", "vperm2f128",
    "vblendpd", "vshufpd"]

function view_asm(@nospecialize(func), @nospecialize(types))

    code_lines = asm_lines(func, types)
    for i = 1 : length(code_lines)
        if (i < length(code_lines) && code_lines[i][1] == "cmp"
                                   && code_lines[i+1][1] in JUMP_TYPES)
            args = split(join(code_lines[i][2:end], ' '), ", ")
            @assert length(args) == 2
            @assert length(code_lines[i+1]) == 2
            prefix = JUMP_PREFIXES[code_lines[i+1][1]]
            symbol = JUMP_SYMBOLS[code_lines[i+1][1]]
            code_lines[i] = [">>>>if (" * prefix * args[1] * symbol * args[2] *
                             ") goto " * code_lines[i+1][2] * ";"]
            code_lines[i+1] = ["nop"]
        end
        if (i < length(code_lines) && code_lines[i][1] == "test"
                                   && code_lines[i+1][1] in JUMP_TYPES)
            args = split(join(code_lines[i][2:end], ' '), ", ")
            @assert length(args) == 2
            @assert length(code_lines[i+1]) == 2
            prefix = JUMP_PREFIXES[code_lines[i+1][1]]
            symbol = JUMP_SYMBOLS[code_lines[i+1][1]]
            if (args[1] == args[2])
                code_lines[i] = [">>>>if (" * prefix * args[1] * symbol *
                                 "0) goto " * code_lines[i+1][2] * ";"]
                code_lines[i+1] = ["nop"]
            end
        end
    end

    unknown_instrs = Dict{String,Int}()
    for line in code_lines

        # Preprocessed lines
        if length(line) == 1 && startswith(line[1], ">>>>")
            say("    ", line[1][5:end])

        # Unprinted lines
        elseif length(line) == 1 && line[1] == ".text"
            # Ignore this line.
        elseif line[1] in ["nop", "push", "pop", "vzeroupper"]
            # Ignore this line.

        # Labels
        elseif length(line) == 1 && endswith(line[1], ':')
            say(line[1])

        # Moves
        elseif line[1] in MOV_TYPES
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 2
            say("    ", line[1], " = ", line[2], ';')
        elseif line[1] == "lea"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 2
            @assert startswith(line[2], '[') && endswith(line[2], ']')
            say("    ", line[1], " = ", line[2][2:end-1], ';')
        elseif line[1] == "seto"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 1
            say("    ", line[1], " = OF;")

        # Increment and decrement
        elseif line[1] == "inc"
            if length(line) == 2
                say("    ++", line[2], ";")
            else
                say("    ++(", join(line[2:end], ' '), ");")
            end
        elseif line[1] == "dec"
            if length(line) == 2
                say("    --", line[2], ";")
            else
                say("    --(", join(line[2:end], ' '), ");")
            end

        # Scalar arithmetic and shifts
        elseif line[1] == "add"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 2
            say("    ", line[1], " += ", line[2], ';')
        elseif line[1] == "imul"
            line = split(join(line[2:end], ' '), ", ")
            if length(line) == 1
                say("    rdx:rax = (signed) ", line[1], " * rax;")
            elseif length(line) == 2
                say("    ", line[1], " *= (signed) ", line[2], ';')
            elseif length(line) == 3
                say("    ", line[1], " = (signed) ", line[2],
                    " * ", line[3], ';')
            else
                @assert false
            end
        elseif line[1] == "sub"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 2
            say("    ", line[1], " -= ", line[2], ';')
        elseif line[1] == "and"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 2
            say("    ", line[1], " &= ", line[2], ';')
        elseif line[1] == "xor"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 2
            if line[1] == line[2]
                say("    ", line[1], " = 0;")
            else
                say("    ", line[1], " ^= ", line[2], ';')
            end
        elseif line[1] == "shl" || line[1] == "sal"
            line = split(join(line[2:end], ' '), ", ")
            if length(line) == 1
                say("    ", line[1], " <<= 1;")
            elseif length(line) == 2
                say("    ", line[1], " <<= ", line[2], ';')
            else
                @assert false
            end
        elseif line[1] == "shr"
            line = split(join(line[2:end], ' '), ", ")
            if length(line) == 1
                say("    ", line[1], " >>>= 1;")
            elseif length(line) == 2
                say("    ", line[1], " >>>= ", line[2], ';')
            else
                @assert false
            end
        elseif line[1] == "sar"
            line = split(join(line[2:end], ' '), ", ")
            if length(line) == 1
                say("    ", line[1], " >>= 1;")
            elseif length(line) == 2
                say("    ", line[1], " >>= ", line[2], ';')
            else
                @assert false
            end
        elseif line[1] == "andn"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 3
            say("    ", line[1], " = ~", line[2], " & ", line[3], ';')


        # Vector arithmetic
        elseif line[1] == "vmovsd"
            line = split(join(line[2:end], ' '), " # ")
            @assert 1 <= length(line) <= 2
            line = split(line[1], ", ")
            @assert length(line) == 2
            say("    ", line[1], " = ", line[2], "; // scalar")
        elseif line[1] == "vaddsd"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 3
            say("    ", line[1], " = ", line[2], " + ", line[3], "; // scalar")
        elseif line[1] == "vaddpd"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 3
            say("    ", line[1], " = ", line[2], " + ", line[3], ';')
        elseif line[1] == "vsubsd"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 3
            say("    ", line[1], " = ", line[2], " - ", line[3], "; // scalar")
        elseif line[1] == "vsubpd"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 3
            say("    ", line[1], " = ", line[2], " - ", line[3], ';')
        elseif line[1] == "vmulsd"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 3
            say("    ", line[1], " = ", line[2], " * ", line[3], "; // scalar")
        elseif line[1] == "vmulpd"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 3
            say("    ", line[1], " = ", line[2], " * ", line[3], ';')
        elseif line[1] == "vdivpd"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 3
            say("    ", line[1], " = ", line[2], " / ", line[3], ';')
        elseif line[1] == "vxorpd"
            line = split(join(line[2:end], ' '), ", ")
            @assert length(line) == 3
            if line[2] == line[3]
                say("    ", line[1], " = 0.0;")
            else
                say("    ", line[1], " = ", line[2], " / ", line[3], ';')
            end
        elseif line[1] in SHUFFLE_INSTRUCTIONS
            line = split(join(line[2:end], ' '), " # ")
            @assert length(line) == 2
            say("    ", replace(line[2], "]," => "], "), ';')

        # Control flow
        elseif line[1] == "jmp"
            @assert length(line) == 2
            say("    goto ", line[2], ';')
        elseif line[1] == "call"
            if length(line) == 2
                say("    ", line[2], "();")
            else
                say("    (", join(line[2:end], ' '), ")();")
            end
        elseif line[1] == "ret"
            say("    return;")
        elseif line[1] == "ud2"
            say("    <unreachable code>")

        # Unknown instructions
        else
            say("    {", join(line, ' '), '}')
            unknown_instrs[line[1]] = get(unknown_instrs, line[1], 0) + 1
        end
    end

    if length(unknown_instrs) > 0
        say()
        freqs = sort!(reverse.(collect(unknown_instrs)), rev=true)
        total = sum(f[1] for f in freqs)
        say("$(total) unknown instructions:")
        for i = 1 : min(length(freqs), 10)
            freq, name = freqs[i]
            say("    ", name, " (", freq, ")")
        end
    end
end

end # module DZMisc
