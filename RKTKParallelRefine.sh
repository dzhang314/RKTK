#!/bin/bash

set -euo pipefail
rktk_dir=$(dirname "$(readlink -f "$0")")
refine_script="$rktk_dir/RKTKRefine.jl"
if [ ! -f "$refine_script" ]; then
    echo "ERROR: File $refine_script does not exist."
    exit 1
fi

num_cores=$(nproc)
num_windows=$((num_cores / 4))
if [ $num_windows -lt 1 ]; then
    num_windows=1
fi
echo "Running $refine_script in parallel using $num_windows windows."

commands=()
for filename in $(ls | sort -r); do
    commands+=("julia -O3 $refine_script $filename")
done
echo "Found ${#commands[@]} files to refine."

tmux new-session -d

for window in $(seq 1 $num_windows); do
    if [ $window -gt 1 ]; then
        tmux new-window
    fi
    tmux split-window -v
    tmux select-pane -t 0
    tmux split-window -h
    tmux select-pane -t 2
    tmux split-window -h
done

index=0
while [ $index -lt ${#commands[@]} ]; do
    available_panes=$(tmux list-panes -a -F '#{pane_id} #{pane_current_command}' | grep 'bash' | awk '{print $1}' || true)
    for pane in $available_panes; do
        echo "Running in pane $pane: ${commands[$index]}"
        tmux send-keys -t "$pane" -l "${commands[$index]}"
        tmux send-keys -t "$pane" Enter
        index=$((index + 1))
        if [ $index -ge ${#commands[@]} ]; then
            break
        fi
    done
    sleep 1
done

echo "Waiting for all commands to finish..."
while true; do
    if [ -z "$(tmux list-panes -a -F '#{pane_current_command}' | grep -v bash)" ]; then
        break
    fi
    sleep 1
done

tmux kill-session
