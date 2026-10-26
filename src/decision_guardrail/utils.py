"""Small serialization and fingerprint helpers."""
import hashlib
import json
from pathlib import Path
from typing import Any


def fingerprint(value: Any) -> str:
    """Hash a JSON-serializable value deterministically."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically publish JSON on the destination filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON."""
    return json.loads(path.read_text(encoding='utf-8'))
