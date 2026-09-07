import json
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def store_raw_event(event: dict) -> None:
    file_path = RAW_DATA_DIR / "simulation.jsonl"

    with file_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")
        