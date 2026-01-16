# src/data.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
GE_DIR = REPO_ROOT / "great_expectations"

DEFAULT_CSV = DATA_DIR / "telco_churn.csv"
TARGET_COL = "Churn"


def load_raw(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing dataset at: {csv_path}\n"
            f"Place Telco dataset as: {DEFAULT_CSV}"
        )
    df = pd.read_csv(csv_path)
    return df


def basic_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal normalization so downstream code is stable.
    - Trim column names
    - Trim string cells
    - Normalize target column to 0/1 if present
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Trim strings
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # Normalize target
    if TARGET_COL in df.columns:
        # common variants: Yes/No, True/False, 1/0 as strings
        mapping = {"Yes": 1, "No": 0, "True": 1, "False": 0, "1": 1, "0": 0}
        df[TARGET_COL] = df[TARGET_COL].map(lambda x: mapping.get(str(x), x))
        if not pd.api.types.is_numeric_dtype(df[TARGET_COL]):
            raise ValueError(
                f"{TARGET_COL} could not be normalized to numeric 0/1. "
                f"Unique values: {df[TARGET_COL].unique()[:10]}"
            )
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    drop_cols = [TARGET_COL]

    # Drop ID-like columns if present
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            drop_cols.append(col)

    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]
    return X, y



# ---------------- Great Expectations helpers ----------------

def _ensure_ge_dir():
    GE_DIR.mkdir(exist_ok=True, parents=True)


def ge_init_suite() -> None:
    """
    Writes a simple data contract file (JSON) that we validate against.
    We keep the function name for convenience, but this is version-proof.
    """
    import json

    _ensure_ge_dir()
    contract_path = GE_DIR / "contract.json"

    contract = {
        "target_column": TARGET_COL,
        "min_columns": 5,
        "allowed_target_values": [0, 1],
        "required_columns": [TARGET_COL],
    }

    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"[Contract] Wrote: {contract_path}")



def ge_validate(df: pd.DataFrame) -> bool:
    """
    Version-proof validation:
    - required columns exist
    - minimum column count
    - target values are only 0/1
    Writes a validation_result.json report.
    """
    import json

    contract_path = GE_DIR / "contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(
            f"Missing contract at {contract_path}. Run: python -m src.data ge-init"
        )

    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    errors = []

    # Column count
    min_cols = int(contract.get("min_columns", 1))
    if df.shape[1] < min_cols:
        errors.append(f"Expected at least {min_cols} columns, got {df.shape[1]}.")

    # Required columns
    required = contract.get("required_columns", [])
    for c in required:
        if c not in df.columns:
            errors.append(f"Missing required column: {c}")

    # Target validation
    target = contract.get("target_column", TARGET_COL)
    if target in df.columns:
        if df[target].isna().any():
            errors.append(f"Target column '{target}' contains nulls.")
        allowed = set(contract.get("allowed_target_values", [0, 1]))
        unique_vals = set(df[target].dropna().unique().tolist())
        if not unique_vals.issubset(allowed):
            errors.append(
                f"Target column '{target}' has unexpected values: {sorted(unique_vals)}. "
                f"Allowed: {sorted(allowed)}"
            )

    ok = len(errors) == 0

    report = {
        "success": ok,
        "errors": errors,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
    }

    report_path = GE_DIR / "validation_result.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Contract] Validation success={ok}. Report: {report_path}")

    if not ok:
        # print the errors to make debugging immediate
        for e in errors:
            print(f"  - {e}")

    return ok




def main():
    """
    Convenience entrypoints:
      python -m src.data ge-init
      python -m src.data ge-validate
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.data ge-init|ge-validate")
        raise SystemExit(2)

    cmd = sys.argv[1].strip().lower()
    if cmd == "ge-init":
        ge_init_suite()
    elif cmd == "ge-validate":
        df = basic_normalize(load_raw())
        ok = ge_validate(df)
        raise SystemExit(0 if ok else 1)
    else:
        print(f"Unknown command: {cmd}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
