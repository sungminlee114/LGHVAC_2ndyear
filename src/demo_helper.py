from pathlib import Path
import json

from src import BASE_DIR

dataset_name = "v7-250309-reduceinputanddatefunctioncall"
BASE_DATASET_DIR = BASE_DIR / "../finetuning/dataset" / dataset_name

def get_available_metadatas():
    scenario_dirs = [d for d in BASE_DATASET_DIR.iterdir() if d.is_dir() and "scenario" in d.name and "metadata.json" in [f.name for f in d.iterdir()]]
    available_metadatas = {}
    for scenario_dir in scenario_dirs:
        metadata_path = scenario_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        available_metadatas[scenario_dir.name] = metadata
    return available_metadatas