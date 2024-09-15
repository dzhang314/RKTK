#!/bin/bash

set -euo pipefail

output=$(julia -O3 RKTKSearch.jl BEM1 6 7 0x17 0x17 --no-file)
hash=$(echo "$output" | md5sum | awk '{print $1}')

if [[ "$hash" == "23d7aa6a0221fa38279740fad7b93818" ]]; then
    echo "Reproducibility test 1 passed."
else
    echo "Reproducibility test 1 failed."
    exit 1
fi

output=$(julia -O3 RKTKSearch.jl BEM1 6 7 0x17 0x17 --no-file --simd)
hash=$(echo "$output" | md5sum | awk '{print $1}')

if [[ "$hash" == "23d7aa6a0221fa38279740fad7b93818" ]]; then
    echo "SIMD Reproducibility test 1 passed."
else
    echo "SIMD Reproducibility test 1 failed."
    exit 10
fi

output=$(julia -O3 RKTKSearch.jl BEM1 7 9 0x01 0x01 --no-file)
hash=$(echo "$output" | md5sum | awk '{print $1}')

if [[ "$hash" == "326e91bc336c090ab19f9a24475ee68d" ]]; then
    echo "Reproducibility test 2 passed."
else
    echo "Reproducibility test 2 failed."
    exit 2
fi

output=$(julia -O3 RKTKSearch.jl BEM1 7 9 0x01 0x01 --no-file --simd)
hash=$(echo "$output" | md5sum | awk '{print $1}')

if [[ "$hash" == "326e91bc336c090ab19f9a24475ee68d" ]]; then
    echo "SIMD Reproducibility test 2 passed."
else
    echo "SIMD Reproducibility test 2 failed."
    exit 20
fi
