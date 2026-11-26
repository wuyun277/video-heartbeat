"""CLI for prepare, train, test and export; no work happens on import."""
import argparse
import logging
from pathlib import Path
from training.settings import DEFAULT_CONFIG, MODEL_NAMES, read_config
from training.data.preprocessing import prepare_subject
from training.engine.trainer import train
from training.engine.evaluator import test
from training.engine.exporter import export_weights

__doc__ = "Commands: prepare, train, test, export. Run from the project root with -m."

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "test", "export"):
        child = sub.add_parser(command)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        child.add_argument("--model", choices=MODEL_NAMES,
                           help="Select training profile; otherwise use active_model in config")
        if command == "train":
            group = child.add_mutually_exclusive_group()
            group.add_argument("--resume", type=Path)
            group.add_argument("--init-weights", type=Path)
            child.add_argument("--stop-after-epoch", type=int)
        if command in ("test", "export"):
            child.add_argument("--checkpoint", required=True, type=Path)
        if command == "export":
            child.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    cfg = read_config(args.config, args.model)
    if args.command == "prepare":
        for split in ("train", "valid", "test"):
            for name in cfg["subjects"][split]:
                prepare_subject(cfg, name)
    elif args.command == "train":
        train(cfg, args)
    elif args.command == "test":
        test(cfg, args)
    else:
        export_weights(cfg, args)
