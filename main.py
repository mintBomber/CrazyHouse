"""
Entry point for the Chess × Shogi hybrid game.

Usage:
    python main.py                  # launch GUI
    python main.py --run run_001    # launch GUI with checkpoints/run_001/latest.pt
    python main.py --ckpt PATH      # launch GUI with a specific checkpoint
"""
from __future__ import annotations

import argparse
import os

import yaml


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_config() -> dict:
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def _default_checkpoint_path(cfg: dict) -> str:
    train_cfg = cfg.get("training", {})
    ckpt_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    run_name = train_cfg.get("run_name", "run_001")
    return _resolve_project_path(os.path.join(ckpt_dir, run_name, "latest.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chess x Shogi - Gumbel AlphaZero")
    parser.add_argument("--run", type=str, default=None,
                        help="Override training.run_name for the GUI checkpoint")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a checkpoint file to load for the GUI AI")
    args = parser.parse_args()

    cfg = _load_config()
    cfg["_project_root"] = PROJECT_ROOT

    if args.run:
        cfg.setdefault("training", {})["run_name"] = args.run

    if args.ckpt is None:
        ckpt_path = _default_checkpoint_path(cfg)
    elif args.ckpt == "":
        ckpt_path = ""
    else:
        ckpt_path = _resolve_project_path(args.ckpt)

    from ui.ui import ChessUI
    ui = ChessUI(cfg, PROJECT_ROOT, default_ckpt=ckpt_path)
    ui.run()


if __name__ == "__main__":
    main()
