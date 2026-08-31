import json
import os
from typing import List, Set


def load_done_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()

    done = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "id" in row:
                done.add(row["id"])

    return done


def load_existing_records(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []

    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records


def open_output(path: str, append: bool = True):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return open(path, "a" if append else "w")