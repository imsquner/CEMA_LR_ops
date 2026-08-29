from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import prepare, run_loop, smoke
from .worker import main as worker_main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Causal filter closed-loop experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "smoke", "run"):
        command = sub.add_parser(name)
        command.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
        command.add_argument("--output", default=None)
    worker = sub.add_parser("worker")
    worker.add_argument("--spec", required=True)
    worker.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "worker":
        return worker_main(["--spec", args.spec, "--output", args.output])
    root = Path(args.root).resolve()
    output = Path(args.output).resolve() if args.output else root / "outputs" / "lr_filter_closed_loop_1h"
    result = {"prepare": prepare, "smoke": smoke, "run": run_loop}[args.command](root, output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

