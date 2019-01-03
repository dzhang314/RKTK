#!/usr/bin/env python3

import datetime
import os
import sys
import time

TASK_DIR = "data"
REQUEST_DIR = "requests"

if os.path.isdir(REQUEST_DIR):
    print("Serving requests in existing \"" +
          REQUEST_DIR + "\" directory.", flush=True)
else:
    try:
        os.mkdir(REQUEST_DIR)
        print("Serving requests in newly-created \"" +
              REQUEST_DIR + "\" directory.", flush=True)
    except FileExistsError:
        print("ERROR: A file named \"" +
              REQUEST_DIR + "\" already exists.", flush=True)
        sys.exit(1)

served = {}
while True:
    # print("Scanning \"" + REQUEST_DIR + "\" directory.")
    requests = {name for name in os.listdir(REQUEST_DIR)
                     if os.path.isdir(os.path.join(REQUEST_DIR, name))}
    # print("Found {0} active requests.".format(len(requests)))
    # There are three stages in the life cycle of a request:
    #     (1) Created: when a requestor creates a new empty request directory.
    #     (2) Served: when we populate its request directory.
    #     (3) Acknowledged: when its requestor deletes its request directory.
    acknowledged = served.keys() - requests
    for name in acknowledged:
        print("Task {0} has been acknowledged.".format(name), flush=True)
        del served[name]
    present = datetime.datetime.now()
    timedout = set()
    for name, timestamp in served.items():
        if present - timestamp > datetime.timedelta(seconds=10):
            print("Task {0} has timed out.".format(name), flush=True)
            for item in os.listdir(os.path.join(REQUEST_DIR, name)):
                os.rename(os.path.join(REQUEST_DIR, name, item),
                          os.path.join(TASK_DIR, item))
            os.rmdir(os.path.join(REQUEST_DIR, name))
            timedout.add(name)
    for name in timedout:
        del served[name]
    created = requests - timedout - served.keys()
    for name in created:
        served[name] = present
        for item in list(os.scandir(TASK_DIR))[::-1]:
            print("Serving request {0} with task {1}.".format(name, item.path), flush=True)
            os.rename(item.path,
                      os.path.join(REQUEST_DIR, name, item.name))
            break
        else:
            print("Serving request {0} with null task.".format(name), flush=True)
            with open(os.path.join(REQUEST_DIR, name, "EXHAUSTED"),
                      'a') as dummyfile:
                pass
    # time.sleep(1)
