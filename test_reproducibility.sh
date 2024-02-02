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
