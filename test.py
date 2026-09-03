import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--GPU', type=int, help="Input the GPU number you'd like to use")
args = parser.parse_args()

print(f"TEST:   GPU Chosen: {args.GPU}")

from src.Tseetup import testup
testup(args.GPU)
