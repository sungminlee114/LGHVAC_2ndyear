from dataclasses import dataclass

@dataclass
class InstructionQ:
    @dataclass
    class QArgs:
        """
        "args": {
            "table_name": "data_t",
            "columns": [
                "oper"
            ],
            "temporal": "[DATE_TRUNC('day', DATE 'CURRENT_DATE' - INTERVAL '1 day'), DATE_TRUNC('day', DATE 'CURRENT_DATE'))",
            "spatials": [
                "01_IB5"
            ]
        }
        """
        table_name: str
        columns: list[str]
        temporal: str
        spatials: list[str]
        metadata: dict
    
    args: QArgs
    result_name: str

@dataclass
class Mapping:
    """
    "args": {
        "temporal": {
            "올해봄": "[DATE 'CURRENT_YEAR-03-01', DATE 'CURRENT_YEAR-06-01')"
        },
        "spatials": {
            "옆반": "01_IB7"
        },
        "modalities": {
            "실내온도": "roomtemp"
        }
    }
    """
    temporal: dict
    spatials: dict
    modalities: dict

@dataclass
class InstructionQ_raw:
    query: str
    result_name: str

@dataclass
class InstructionO:
    scripts: list[str]

@dataclass
class InstructionG:
    axes: list[dict]
    required_variables: list[str]


@dataclass
class InstructionR:
    expectations: list[str]
    required_variables: list[str]