import random
import sys
import os
from datetime import datetime
from pathlib import Path

import torch
from argparse import Namespace


def set_deterministic_inference(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CliLogger:
    LEVEL_ORDER = {"debug": 10, "info": 20, "warn": 30, "error": 40}
    RESET = "\033[0m"
    COLORS = {
        "debug": "\033[36m",
        "info": "\033[34m",
        "warn": "\033[33m",
        "error": "\033[31m",
        "success": "\033[32m",
        "section": "\033[35m",
        "protein": "\033[96m",
    }

    def __init__(self, level="info", use_color=True):
        self.level = level if level in self.LEVEL_ORDER else "info"
        self.use_color = use_color

    def _allowed(self, level):
        return self.LEVEL_ORDER.get(level, 20) >= self.LEVEL_ORDER[self.level]

    def _paint(self, text, key):
        if not self.use_color:
            return text
        return f"{self.COLORS.get(key, '')}{text}{self.RESET}"

    def log(self, level, message, protein=None):
        if not self._allowed(level):
            return
        ts = datetime.now().strftime("%H:%M:%S")
        level_tag = self._paint(level.upper().ljust(5), level)
        protein_tag = ""
        if protein:
            protein_tag = f" {self._paint(f'[{protein}]', 'protein')}"
        print(f"[{ts}] {level_tag}{protein_tag} {message}")

    def debug(self, message, protein=None):
        self.log("debug", message, protein=protein)

    def info(self, message, protein=None):
        self.log("info", message, protein=protein)

    def warn(self, message, protein=None):
        self.log("warn", message, protein=protein)

    def error(self, message, protein=None):
        self.log("error", message, protein=protein)

    def success(self, message, protein=None):
        ts = datetime.now().strftime("%H:%M:%S")
        tag = self._paint("OK   ", "success")
        protein_tag = ""
        if protein:
            protein_tag = f" {self._paint(f'[{protein}]', 'protein')}"
        print(f"[{ts}] {tag}{protein_tag} {message}")

    def section(self, title):
        divider = "=" * 72
        if self.use_color:
            divider = self._paint(divider, "section")
            title = self._paint(title, "section")
        print(divider)
        print(title)
        print(divider)


def should_use_color(no_color=False):
    if no_color:
        return False
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def read_names(fasta_dir):
    files = Path(fasta_dir).glob("*.fasta")
    names = [file.stem for file in files]
    return names


def clone_args_with_overrides(args, **kwargs):
    copied = Namespace(**vars(args))
    for key, value in kwargs.items():
        setattr(copied, key, value)
    return copied


def format_name_preview(names, max_items=8):
    if len(names) <= max_items:
        return ", ".join(names)
    head_n = max(1, max_items // 2)
    tail_n = max_items - head_n
    head = ", ".join(names[:head_n])
    tail = ", ".join(names[-tail_n:])
    return f"{head}, ..., {tail}"


def _fit_text(text, width):
    text = str(text)
    if len(text) <= width:
        return text.ljust(width)
    return (text[: width - 1] + "…") if width > 1 else text[:width]


def print_compare_table_header(
    logger,
    include_venus=True,
    raw_label="Raw",
    venus_label="VenusREM",
    orbit_label="Orbit",
    orbit_cal_label=None,
    current_label="Current",
):
    protein_w = 40
    metric_w = 9
    delta_w = 11
    if include_venus and orbit_cal_label is not None:
        raw_h = _fit_text(raw_label, metric_w)
        venus_h = _fit_text(venus_label, metric_w)
        orbit_h = _fit_text(orbit_label, metric_w)
        orbit_cal_h = _fit_text(orbit_cal_label, metric_w)
        delta_orbit_h = _fit_text("d_orbit-v1", delta_w)
        delta_cal_h = _fit_text("d_orbit-cal-v1", delta_w)
        header = (
            f"| {'Protein'.ljust(protein_w)} | {raw_h.rjust(metric_w)} | "
            f"{venus_h.rjust(metric_w)} | {orbit_h.rjust(metric_w)} | "
            f"{orbit_cal_h.rjust(metric_w)} | {delta_orbit_h.rjust(delta_w)} | "
            f"{delta_cal_h.rjust(delta_w)} |"
        )
        bar = (
            f"+-{'-' * protein_w}-+-{'-' * metric_w}-+-{'-' * metric_w}-+-{'-' * metric_w}-"
            f"+-{'-' * metric_w}-+-{'-' * delta_w}-+-{'-' * delta_w}-+"
        )
    elif include_venus:
        raw_h = _fit_text(raw_label, metric_w)
        venus_h = _fit_text(venus_label, metric_w)
        orbit_h = _fit_text(orbit_label, metric_w)
        header = (
            f"| {'Protein'.ljust(protein_w)} | {raw_h.rjust(metric_w)} | "
            f"{venus_h.rjust(metric_w)} | {orbit_h.rjust(metric_w)} | {'Delta'.rjust(delta_w)} |"
        )
        bar = (
            f"+-{'-' * protein_w}-+-{'-' * metric_w}-+-{'-' * metric_w}-+-{'-' * metric_w}-+-{'-' * delta_w}-+"
        )
    else:
        raw_h = _fit_text(raw_label, metric_w)
        current_h = _fit_text(current_label, metric_w)
        header = (
            f"| {'Protein'.ljust(protein_w)} | {raw_h.rjust(metric_w)} | "
            f"{current_h.rjust(metric_w)} | {'Delta'.rjust(delta_w)} |"
        )
        bar = f"+-{'-' * protein_w}-+-{'-' * metric_w}-+-{'-' * metric_w}-+-{'-' * delta_w}-+"
    logger.info(bar)
    logger.info(header)
    logger.info(bar)


def print_compare_table_row(logger, protein_name, raw_corr, orbit_corr, venus_corr=None):
    return print_compare_table_row_extended(
        logger=logger,
        protein_name=protein_name,
        raw_corr=raw_corr,
        orbit_corr=orbit_corr,
        venus_corr=venus_corr,
        orbit_cal_corr=None,
    )


def print_compare_table_row_extended(
    logger,
    protein_name,
    raw_corr,
    orbit_corr,
    venus_corr=None,
    orbit_cal_corr=None,
):
    protein_w = 40
    metric_w = 9
    delta_w = 11
    raw_s = f"{raw_corr:.4f}".rjust(metric_w)
    orbit_s = f"{orbit_corr:.4f}".rjust(metric_w)
    protein_s = _fit_text(protein_name, protein_w)
    if venus_corr is not None and orbit_cal_corr is not None:
        venus_s = f"{venus_corr:.4f}".rjust(metric_w)
        orbit_cal_s = f"{orbit_cal_corr:.4f}".rjust(metric_w)
        delta_orbit = orbit_corr - venus_corr
        delta_orbit_s = f"{delta_orbit:+.4f}".rjust(delta_w)
        delta_cal = orbit_cal_corr - venus_corr
        delta_cal_s = f"{delta_cal:+.4f}".rjust(delta_w)
        if logger.use_color:
            delta_orbit_color = "success" if delta_orbit >= 0 else "error"
            delta_orbit_s = logger._paint(delta_orbit_s, delta_orbit_color)
            delta_cal_color = "success" if delta_cal >= 0 else "error"
            delta_cal_s = logger._paint(delta_cal_s, delta_cal_color)
        row = (
            f"| {protein_s} | {raw_s} | {venus_s} | {orbit_s} | "
            f"{orbit_cal_s} | {delta_orbit_s} | {delta_cal_s} |"
        )
    elif venus_corr is not None:
        venus_s = f"{venus_corr:.4f}".rjust(metric_w)
        delta = orbit_corr - venus_corr
        delta_s = f"{delta:+.4f}".rjust(delta_w)
        if logger.use_color:
            delta_color = "success" if delta >= 0 else "error"
            delta_s = logger._paint(delta_s, delta_color)
        row = f"| {protein_s} | {raw_s} | {venus_s} | {orbit_s} | {delta_s} |"
    else:
        delta = orbit_corr - raw_corr
        delta_s = f"{delta:+.4f}".rjust(delta_w)
        if logger.use_color:
            delta_color = "success" if delta >= 0 else "error"
            delta_s = logger._paint(delta_s, delta_color)
        row = f"| {protein_s} | {raw_s} | {orbit_s} | {delta_s} |"
    logger.info(row)
