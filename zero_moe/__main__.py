import argparse
import json
import math
import os
from pathlib import Path
from zipfile import BadZipFile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from .checkpoint import pack_checkpoint
from .experiment import evaluate, predict, train
from .model import EXPERT_COUNT


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def seed_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("seed must be nonnegative")
    return value


def time_limit(value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value <= 30.0:
        raise argparse.ArgumentTypeError("minutes must be in (0, 30]")
    return value


def main():
    parser = argparse.ArgumentParser(description="ZeroNet-2B-MoE")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train experts and save FP16 weights")
    training.add_argument("--weights", type=Path, default=Path("weights"))
    training.add_argument("--experts", type=positive_int, default=EXPERT_COUNT)
    training.add_argument("--steps", type=positive_int, default=64)
    training.add_argument("--seed", type=seed_int, default=2026)
    training.add_argument("--minutes", type=time_limit, default=30.0)

    packing = commands.add_parser("pack", help="pack trained weights into an INT8 release ZIP")
    packing.add_argument("--weights", type=Path, default=Path("weights"), help="training directory")
    packing.add_argument("--output", type=Path, required=True, help="new ZIP file")

    evaluation = commands.add_parser("evaluate", help="score saved weights on held-out data")
    evaluation.add_argument("--weights", type=Path, default=Path("weights"), help="directory or release ZIP")
    evaluation.add_argument("--cases", type=positive_int, default=256, help="cases per expert")
    evaluation.add_argument("--seed", type=seed_int, default=2027)

    inference = commands.add_parser("predict", help="classify numbers provided as separate arguments")
    inference.add_argument("--weights", type=Path, default=Path("weights"), help="directory or release ZIP")
    inference.add_argument("--seed", type=seed_int, help="seed the random expert routing")
    inference.add_argument("numbers", type=float, nargs="+")

    args = parser.parse_args()
    try:
        if args.command == "train":
            if args.experts > EXPERT_COUNT:
                parser.error(f"--experts cannot exceed {EXPERT_COUNT}")
            metrics = train(args.weights, args.experts, args.steps, args.seed, args.minutes)
        elif args.command == "pack":
            metrics = pack_checkpoint(args.weights, args.output)
        elif args.command == "evaluate":
            metrics = evaluate(args.weights, args.cases, args.seed)
        else:
            for label in predict(args.weights, args.numbers, args.seed):
                print("nonzero" if label else "zero")
            return
    except FileNotFoundError as exc:
        parser.exit(1, f"error: missing file {exc.filename}. Check the weights and output paths.\n")
    except FileExistsError as exc:
        parser.exit(1, f"error: {exc.filename} already exists. Choose a new path.\n")
    except (OSError, ValueError, BadZipFile) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
