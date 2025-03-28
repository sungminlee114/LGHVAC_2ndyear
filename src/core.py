__all__ = [
    "load_models",
    "wait_for_input_from_user",
    "input_to_instruction_set",
    # "execute_query",
    # "execute_response_generation",
    # "execute_instruction_set",
    "execute_instruction_set_web",
]

import logging
import pprint
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch

from src.db.manager import DBManager
from src.input_to_instructions.load_and_execute import *
from src.input_to_instructions.types import *
from src.operation.execute import *
from src.plot_graph.execute import *
from src.response_generation.load_and_execute import ResponseGeneration


def load_models():

    InputToInstruction.initialize(
        train_type="ours",
        dtype=["Q8_0", "F16"][1],
        log_output=False
    )

    ResponseGeneration.initialize(
        log_output=False
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def wait_for_input_from_user():
    return input("Enter your query: ")

def input_to_instruction_set(user_input, current_metadata):
    return InputToInstruction.execute(user_input, current_metadata)

# def execute_response_generation(instruction:Instruction, query_mapping : list[list[str, str]], user_input:str, current_metadata:dict):
#     """
#     Generate a response based on the provided instruction.
    
#     instruction: An Instruction object containing the response generation details.
#     query_mapping (list[list[str, str]]): A mapping that associates SQL query results with their corresponding variable names.
#     user_input: The user's input, which may influence the response.
#     current_metadata: Additional metadata that may be relevant to response generation.
    
#     Returns:
#         - response (str): The generated response.
#     """
#     for k,v in query_mapping :
#         if v is None or len(v)==0 :
#             response = "죄송합니다. 해당 정보를 찾을 수 없습니다. (이유 설명 필요)"
#         else:
#             response = ResponseGeneration.execute(instruction, query_mapping, user_input, current_metadata)
        
#     return response

# def execute_instruction_set(semantic:Semantic,instruction_set:list[Instruction], user_input:str, current_metadata:dict, response_function:Callable):
#     """
#     Implement the agent to execute a set of instructions.
    
#     semantic: The semantic information of the input text.
#     instruction_set: A list of Instruction objects to be executed sequentially.
#     user_input: The user's input, which may influence the execution flow.
#     current_metadata: Additional metadata that may be relevant to the execution.
#     """
#     query_mapping = []
#     for instruction in instruction_set:
#         logger.info(f"Executing instruction: {instruction}")
        
#         if type(instruction) == InstructionQ:
#             execute_query(instruction)
#             query_mapping.append([instruction.save_variable,instruction.value])
            
#         elif instruction.operation_flag == "r":
#             pass
#             # Execute response generation
#             response = execute_response_generation(instruction, query_mapping, user_input, current_metadata)
#             response_function(response)

def execute_instruction_set_web(instructions:list[InstructionQ|InstructionO|InstructionR|None], user_input:str, metadata:dict, response_function:Callable):
    """
    This function is a duplicate of execute_instruction_set, but with additional support for web-based responses.
    """
    variables = {
        "Metadata": metadata,
    }
    for instruction in instructions:
        logger.debug(f"Executing instruction: {instruction.__class__.__name__}")
        yield from response_function(f"Executing instruction: {instruction.__class__.__name__}")
        
        if type(instruction) == InstructionQ:
            # Execute query
            result_df = DBManager.structured_query_data_t(metadata, instruction.args)
            if result_df is None:
                yield from response_function("죄송합니다, 관련 데이터를 찾을 수 없습니다.", "response")
                return

            # For demo, drop rows where any value is -1
            result_df = result_df.loc[(result_df != -1).all(axis=1)]
            yield from response_function(f"QueryResult: {result_df}")

            variables[instruction.result_name] = result_df
        
        elif type(instruction) == InstructionO:
            # Execute operation

            result_dict = OperationExecutor.execute(variables, instruction.scripts, instruction.returns)
            variables.update(result_dict)
            pass
        elif type(instruction) == InstructionG:

            # fig_html = plot_graph(instruction, variables, return_html=True)
            fig_html = plot_graph_plotly(instruction, variables, return_html=True)
            
            # yield from response_function("<h2 Generated graph:>")
            yield from response_function(fig_html, "graph")
        elif type(instruction) == InstructionR:
            # Execute response generation
            variables_to_report = {k: v for k, v in variables.items() if k not in ["Metadata"]}
            yield from response_function(f"Variables: {variables_to_report}")
            response, required_variables = ResponseGeneration.execute(instruction, variables, user_input, metadata)
            yield from response_function(f"Required variables: {required_variables}")
            
            yield from response_function(response, "response")
            