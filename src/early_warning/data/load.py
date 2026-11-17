"""Raw-data loading with schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from early_warning.data.schema import RAW_DTYPES, REQUIRED_COLUMNS


class SchemaError(ValueError):
    """Raised when the raw file does not match the expected contract."""


def validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")


def load_raw(path: str | Path, sep: str = ";") -> pd.DataFrame:
    """Read the challenge CSV with explicit dtypes (fails fast on schema drift)."""
    header = pd.read_csv(path, sep=sep, nrows=0)
    validate_columns(header)
    return pd.read_csv(path, sep=sep, dtype=RAW_DTYPES)
