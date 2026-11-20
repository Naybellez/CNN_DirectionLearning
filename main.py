# This is a wrapper that calls setup.py from the src folder
# This script allows for multiple simulation to be run using a single bash command
# eg. python3 main.py --GPU 0

# last edited 201126 - tidying up

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--GPU', type=int, help="Input the GPU number you'd like to use")

args = parser.parse_args()

print(f"MAIN:  GPU Chosen: {args.GPU}")

from src.Setup import setup
setup(args.GPU)


