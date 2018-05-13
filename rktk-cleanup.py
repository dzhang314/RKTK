#!/usr/bin/env python3

import glob
import os

filenames = glob.glob("????-????-RKTK-????????-????-????"
                      "-????-????????????-????????????.*")
groups = {}
for filename in filenames:
    key = filename[15:51]
    if key in groups:
        groups[key].add(filename)
    else:
        groups[key] = {filename}

count = 0
for group in groups.values():
    latest = max(group, key=lambda filename: int(filename[52:64]))
    for filename in group:
        if filename != latest:
            os.remove(filename)
            count += 1

print("Deleted {0} redundant files.".format(count))
