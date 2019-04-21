iszero(x::Quad{T}) where {T <: IEEEFloat} =
    iszero(x.x0) & iszero(x.x1) & iszero(x.x2) & iszero(x.x3)
isone(x::Quad{T}) where {T <: IEEEFloat} =
    isone(x.x0) & iszero(x.x1) & iszero(x.x2) & iszero(x.x3)

<(a::Quad{T}, b::Quad{T}) where {T <: AbstractFloat} =
    ((a.x0 < b.x0) |
        ((a.x0 == b.x0) & ((a.x1 < b.x1) |
            ((a.x1 == b.x1) & ((a.x2 < b.x2) |
                ((a.x2 == b.x2) & (a.x3 < b.x3)))))))

<=(a::Quad{T}, b::Quad{T}) where {T <: AbstractFloat} =
    ((a.x0 < b.x0) |
        ((a.x0 == b.x0) & ((a.x1 < b.x1) |
            ((a.x1 == b.x1) & ((a.x2 < b.x2) |
                ((a.x2 == b.x2) & (a.x3 <= b.x3)))))))

iszero(x::Triple{T}) where {T <: IEEEFloat} =
    iszero(x.x0) & iszero(x.x1) & iszero(x.x2)
isone(x::Triple{T}) where {T <: IEEEFloat} =
    isone(x.x0) & iszero(x.x1) & iszero(x.x2)

<(a::Triple{T}, b::Triple{T}) where {T <: AbstractFloat} =
    ((a.x0 < b.x0) |
        ((a.x0 == b.x0) & ((a.x1 < b.x1) |
            ((a.x1 == b.x1) & (a.x2 < b.x2)))))

<=(a::Triple{T}, b::Triple{T}) where {T <: AbstractFloat} =
    ((a.x0 < b.x0) |
        ((a.x0 == b.x0) & ((a.x1 < b.x1) |
            ((a.x1 == b.x1) & (a.x2 <= b.x2)))))
