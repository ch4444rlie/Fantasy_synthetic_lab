"""Thin mission runners used by the modular lab notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .notebook_bridge import run_mission_1 as _run_m1
from .notebook_bridge import run_mission_2 as _run_m2
from .notebook_bridge import run_mission_3 as _run_m3
from .notebook_bridge import run_mission_4 as _run_m4


def run_mission_1() -> Dict[str, Any]:
    return _run_m1()


def run_mission_2(csv_path: str = "mission1_synthetic_data.csv"):
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run Mission 1 first to generate synthetic data."
        )
    return _run_m2(csv_path)


def run_mission_3(csv_path: str = "mission1_synthetic_data.csv"):
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run Mission 1 first to generate synthetic data."
        )
    return _run_m3(csv_path)


def run_mission_4(
    transaction_csv: str = "mission1_synthetic_data.csv",
    docs_folder: str = "mission2_documents",
):
    if not Path(transaction_csv).exists():
        raise FileNotFoundError(
            f"{transaction_csv} not found. Run Mission 1 first to generate synthetic data."
        )
    if not Path(docs_folder).exists():
        raise FileNotFoundError(
            f"{docs_folder} not found. Run Mission 2 first to generate investigation documents."
        )
    return _run_m4(transaction_csv=transaction_csv, docs_folder=docs_folder)
