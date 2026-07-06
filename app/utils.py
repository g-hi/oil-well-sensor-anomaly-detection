from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def setup_logging(name: str = "oil_well_api") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


logger = setup_logging()


class PredictionError(RuntimeError):
    """Raised when inference cannot be completed safely."""


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Unable to read JSON artifact %s: %s", path, exc)
        return None
