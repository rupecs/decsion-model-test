"""Publish an explicitly authored v14 pack only after all construction gates pass."""
import argparse
import json
from pathlib import Path

from decision_guardrail.data_generation.context_v14_release import ConstructionBlocked, build, verify

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authored", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.verify and args.authored is None:
        parser.error("An explicit authored pack is required; this command does not generate filler")
    if (args.authored is not None and not args.authored.resolve().is_relative_to(root)) or not args.output.resolve().is_relative_to(root):
        parser.error("Authored packs and releases must remain inside the project")
    try:
        result = verify(root, args.output) if args.verify else build(root, args.authored, args.output)
    except ConstructionBlocked as error:
        print(json.dumps({k: error.report[k] for k in ("records", "construction_gate_passed", "failures")}, indent=2))
        raise SystemExit(1)
    print(json.dumps(dict(records=result["records"], construction_gate_passed=True, training_ready=False)))
