#!/usr/bin/env python3

import os
import subprocess
import sys
import uuid

REQUEST_DIR = "requests"
OUTPUT_DIR = "refined"

WORKER_ID = str(uuid.uuid4()).upper()
WORKER_DIR = "WORKER-" + WORKER_ID
print("Initializing RKTK refinement worker with ID {0}.".format(WORKER_ID))
os.mkdir(WORKER_DIR)

if os.path.isdir(REQUEST_DIR):
    print("Posting requests in existing \"" +
          REQUEST_DIR + "\" directory.")
else:
    try:
        os.mkdir(REQUEST_DIR)
        print("Posting requests in newly-created \"" +
              REQUEST_DIR + "\" directory.")
    except FileExistsError:
        print("ERROR: A file named \"" +
              REQUEST_DIR + "\" already exists.")
        sys.exit(1)

if os.path.isdir(OUTPUT_DIR):
    print("Sending output to existing \"" +
          OUTPUT_DIR + "\" directory.")
else:
    try:
        os.mkdir(OUTPUT_DIR)
        print("Sending output to newly-created \"" +
              OUTPUT_DIR + "\" directory.")
    except FileExistsError:
        print("ERROR: A file named \"" +
              OUTPUT_DIR + "\" already exists.")
        sys.exit(1)

def work():
    REQUEST_ID = str(uuid.uuid4()).upper()
    REQUEST_NAME = "REQUEST-" + REQUEST_ID
    os.mkdir(os.path.join(REQUEST_DIR, REQUEST_NAME))
    served = False
    while True:
        for entry in os.scandir(os.path.join(REQUEST_DIR, REQUEST_NAME)):
            INPUT_FILE = entry.name
            os.rename(entry.path,
                      os.path.join(WORKER_DIR, INPUT_FILE))
            served = True
        if served:
            break
    os.rmdir(os.path.join(REQUEST_DIR, REQUEST_NAME))
    if INPUT_FILE == "EXHAUSTED":
        return False
    subprocess.run(['../bin/rksearch', '53', '0.5', '0', INPUT_FILE],
                   cwd=WORKER_DIR, stderr=subprocess.DEVNULL)
    subprocess.run(['python3', '../rktk-cleanup.py'],
                   cwd=WORKER_DIR)
    for entry in os.scandir(WORKER_DIR):
        os.rename(entry.path,
                  os.path.join(OUTPUT_DIR, entry.name))
    return True

while True:
    if not work(): break

os.remove(os.path.join(WORKER_DIR, "EXHAUSTED"))
os.rmdir(WORKER_DIR)