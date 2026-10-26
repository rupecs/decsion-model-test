"""Local numeric training telemetry without customer text or network services."""
import json
from pathlib import Path
import time


def append_metric(path: Path, step: int, max_steps: int, epoch, logs: dict) -> None:
    """Append one flushed JSON line; readers may ignore a partial final line."""
    metrics = {k:v for k,v in logs.items() if type(v) in (int, float)}
    event = {'timestamp':time.time(), 'step':step, 'max_steps':max_steps,
             'epoch':epoch, 'metrics':metrics}
    encoded = json.dumps(event, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(encoded+'\n')
