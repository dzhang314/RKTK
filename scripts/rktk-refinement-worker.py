#!/usr/bin/env python3

import os
import subprocess
import sys
import uuid

REQUEST_DIR = "requests"
OUTPUT_DIR = "refined"

RKSEARCH_EXE_PATH = "C:/Users/Zhang/Documents/GitHub/rktk/bin/rksearch-gcc.exe"
PYTHON_EXE_PATH = "C:/Program Files (x86)/IntelSWTools/intelpython3/python.exe"
CLEANUP_SCRIPT_PATH = "C:/Users/Zhang/Documents/GitHub/rktk/scripts/rktk-cleanup.py"

WORKER_ID = str(uuid.uuid4()).upper()
WORKER_DIR = "WORKER-" + WORKER_ID
WORKER_LOG = open("WORKER-" + WORKER_ID + ".log", 'w+')

def log(message):
    print(message, flush=True)
    print(message, file=WORKER_LOG, flush=True)

log("Initializing RKTK refinement worker with ID {0}.".format(WORKER_ID))
os.mkdir(WORKER_DIR)

if os.path.isdir(REQUEST_DIR):
    log("Posting requests in existing \"" + REQUEST_DIR + "\" directory.")
else:
    try:
        os.mkdir(REQUEST_DIR)
        log("Posting requests in new \"" + REQUEST_DIR + "\" directory.")
    except FileExistsError:
        log("ERROR: A file named \"" + REQUEST_DIR + "\" already exists.")
        sys.exit(1)

if os.path.isdir(OUTPUT_DIR):
    log("Sending output to existing \"" + OUTPUT_DIR + "\" directory.")
else:
    try:
        os.mkdir(OUTPUT_DIR)
        log("Sending output to new \"" + OUTPUT_DIR + "\" directory.")
    except FileExistsError:
        log("ERROR: A file named \"" + OUTPUT_DIR + "\" already exists.")
        sys.exit(1)

def work():
    REQUEST_ID = str(uuid.uuid4()).upper()
    REQUEST_NAME = "REQUEST-" + REQUEST_ID
    os.mkdir(os.path.join(REQUEST_DIR, REQUEST_NAME))
    log("Posted request {0}.".format(REQUEST_NAME))
    served = False
    while True:
        for entry in os.scandir(os.path.join(REQUEST_DIR, REQUEST_NAME)):
            INPUT_FILE = entry.name
            os.rename(entry.path, os.path.join(WORKER_DIR, INPUT_FILE))
            served = True
        if served:
            log("Request {0} has been served.".format(REQUEST_NAME))
            break
    os.rmdir(os.path.join(REQUEST_DIR, REQUEST_NAME))
    if INPUT_FILE == "EXHAUSTED":
        log("Null request received. Terminating worker.")
        return False
    log("Running RKSearch refinement job.")
    subprocess.run([RKSEARCH_EXE_PATH, '9', '12', '128', '1', '10', INPUT_FILE],
                   cwd=WORKER_DIR, stderr=subprocess.DEVNULL)
    log("Running cleanup script.")
    subprocess.run([PYTHON_EXE_PATH, CLEANUP_SCRIPT_PATH], cwd=WORKER_DIR)
    log("Submitting job results.")
    for entry in os.scandir(WORKER_DIR):
        os.rename(entry.path, os.path.join(OUTPUT_DIR, entry.name))
    log("Job completed.")
    return True

while True:
    if not work(): break

os.remove(os.path.join(WORKER_DIR, "EXHAUSTED"))
os.rmdir(WORKER_DIR)
WORKER_LOG.close()
