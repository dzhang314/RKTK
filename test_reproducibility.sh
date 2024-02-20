#!/bin/bash

set -euo pipefail

output=$(julia -O3 RKTKSearch.jl 10 16 0x093C 0x093C --no-file)
hash=$(echo "$output" | md5sum | awk '{print $1}')

# Compare the computed hash with the expected hash
if [[ "$hash" == "1ae2777f16e4fd3e5a5034e53f254ef2" ]]; then
    echo "Reproducibility test passed."
else
    echo "Reproducibility test failed."
    exit 1
fi

output=$(julia -O3 RKTKProjectionSearch.jl 7 9 0x001A 0x001A --no-file)
hash=$(echo "$output" | md5sum | awk '{print $1}')

# Compare the computed hash with the expected hash
if [[ "$hash" == "694025855db63f8d22318cfd263bcd59" ]]; then
    echo "Projection reproducibility test passed."
else
    echo "Projection reproducibility test failed."
    exit 1
fi
