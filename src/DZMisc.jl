module DZMisc

export rmk, say, orthonormalize_columns!, linearly_independent_column_indices!

using Base.Threads: lock, unlock, SpinLock
using InteractiveUtils: _dump_function

################################################################################

const DZMISC_STDOUT_LOCK = SpinLock()

function rmk(args...; verbose::Bool=true)::Nothing
    if verbose
        lock(DZMISC_STDOUT_LOCK)
        print("\33[2K\r")
        print(args...)
        flush(stdout)
        unlock(DZMISC_STDOUT_LOCK)
    end
end

function say(args...; verbose::Bool=true)::Nothing
    if verbose
        lock(DZMISC_STDOUT_LOCK)
        print("\33[2K\r")
        println(args...)
        flush(stdout)
        unlock(DZMISC_STDOUT_LOCK)
    end
end

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
    "vblendpd", "vshufpd", "vpermpd", "vpermilpd"]

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
