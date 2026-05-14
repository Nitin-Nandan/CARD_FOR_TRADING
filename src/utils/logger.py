"""
Run logger for CARD training.

Design principle (project rule — applies to ALL agents and scripts):
  - Terminal shows ONLY clean, human-readable progress lines.
  - ALL detail (timestamps, metrics, loss components) goes to the log file only.
  - No WandB, TensorBoard, or dead stubs.
"""

import logging
import re
from pathlib import Path


def _strip_ansi(msg: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", msg)


class RunLogger:
    """
    Two-channel logger:
        file    — DEBUG+, full timestamps (everything)
        console — WARNING+ only (errors and warnings visible at terminal)

    Use print() for clean human-readable terminal output (progress summaries).
    Use logger.info() for detail that belongs only in the log file.
    """

    def __init__(self, log_dir: str, run_name: str):
        self.run_name = run_name
        log_path = Path(log_dir) / f"{run_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path

        self._logger = logging.getLogger(f"card.{run_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        if self._logger.hasHandlers():
            self._logger.handlers.clear()

        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        self._logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
        self._logger.addHandler(ch)

    def info(self, msg: str) -> None:
        """Log to file only — not terminal."""
        self._logger.info(_strip_ansi(msg))

    def warning(self, msg: str) -> None:
        """Log to file AND terminal."""
        self._logger.warning(_strip_ansi(msg))

    def error(self, msg: str) -> None:
        """Log to file AND terminal."""
        self._logger.error(_strip_ansi(msg))

    def metric(self, step: int, **kwargs) -> None:
        """Log numeric metrics to file (one line per call)."""
        parts = "  ".join(
            f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in kwargs.items()
        )
        self._logger.info(f"[step={step}]  {parts}")

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Compatibility shim for older call sites."""
        self.metric(step, **metrics)
