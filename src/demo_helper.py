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
        suggested_inputs_path = scenario_dir / "suggested_inputs.txt"
        with open(suggested_inputs_path, "r") as f:
            suggested_inputs = f.read().splitlines()

        metadata_repr  = [
            ['Site 정보',
                [
                    # ["Site 이름", v.get('site_name', None)],
                    ["Modality 매핑", [f"{k}: {v}" for k, v in metadata.get('modality_mapping', {}).items()]],
                ]
            ],
            ['유저 정보', [
                ["이름", metadata.get('user_name', None)],
                ["역할", metadata.get('user_role', None)],
                ["IDU 이름", metadata.get('idu_name', None)],
                ["IDU 매핑", [f"{k}: {v}" for k, v in metadata.get('idu_mapping', {}).items()]],
            ]],
            ['현재 정보', [
                ["일시", metadata.get('current_datetime', None),] 
            ]],
        ]

        available_metadatas[scenario_dir.name] = {
            "metadata": metadata,
            "metadata_repr": metadata_repr,
            "suggested_inputs": suggested_inputs
        }
    return available_metadatas