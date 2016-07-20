import sys
import argparse
import numpy as np

import ray

parser = argparse.ArgumentParser(description="Parse addresses for the worker to connect to.")
parser.add_argument("--scheduler-address", default="127.0.0.1:10001", type=str, help="the scheduler's address")
parser.add_argument("--objstore-address", default="127.0.0.1:20001", type=str, help="the objstore's address")
parser.add_argument("--worker-address", default="127.0.0.1:40001", type=str, help="the worker's address")

if __name__ == "__main__":
  args = parser.parse_args()
  ray.worker.connect(args.scheduler_address, args.objstore_address, args.worker_address)

  ray.worker.main_loop()
