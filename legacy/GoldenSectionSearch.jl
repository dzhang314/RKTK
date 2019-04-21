module GoldenSectionSearch

export golden_section_search

Base.@irrational INV_PHI    0.61803398874989484820  (sqrt(big(5)) - 1) / 2
Base.@irrational INV_PHI_2  0.38196601125010515180  (3 - sqrt(big(5))) / 2

function golden_section_search(f, a, b, n, args...)
    h = b - a
    c = a + INV_PHI_2 * h
    d = a + INV_PHI * h
    yc = f(c, args...)
    yd = f(d, args...)
    for i = 1 : n
        if yc < yd
            b = d
            d = c
            yd = yc
            h *= INV_PHI
            c = a + INV_PHI_2 * h
            yc = f(c, args...)
        else
            a = c
            c = d
            yc = yd
            h *= INV_PHI
            d = a + INV_PHI * h
            yd = f(d, args...)
        end
    end
    if yc < yd
        c, yc
    else
        d, yd
    end
end

end # module GoldenSectionSearch
