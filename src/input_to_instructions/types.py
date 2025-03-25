__all__ = ['InstructionQ', 'InstructionO', 'InstructionG', 'InstructionR']
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
class InstructionO:
    scripts: list[str]
    returns: list[str]

@dataclass
class InstructionG:
    type: str
    axis: dict
    plots: list[dict]
    required_variables: list[str]


@dataclass
class InstructionR:
    expectations: list[str]
    required_variables: list[str]