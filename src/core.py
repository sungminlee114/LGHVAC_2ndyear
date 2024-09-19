__all__ = [
    "load_models",
    "get_current_metadata",
    "wait_for_input_from_user",
    "input_to_instruction_set",
    "execute_query",
    "execute_response_generation",
    "execute_instruction_set",
]

import logging
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.db.manager import DBManager
from src.input_to_instructions.load_and_execute import InputToInstruction, Instruction, Semantic
from src.instruction_to_sql.load_and_execute import InstructionToSql
from src.response_generation.generate_response import ResponseGeneration


def load_models():
    tokenizer, model, terminators = load_text_model()
    InputToInstruction.model = model
    InputToInstruction.tokenizer = tokenizer
    InputToInstruction.terminators = terminators
    
    ResponseGeneration.model = model
    ResponseGeneration.tokenizer = tokenizer
    ResponseGeneration.terminators = terminators
    
    tokenizer, model = load_sql_model()
    InstructionToSql.model = model
    InstructionToSql.tokenizer = tokenizer
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def load_text_model():
    model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir="/model"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/model"
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    return tokenizer, model, terminators
    

def load_sql_model():
    model_name = "defog/llama-3-sqlcoder-8b"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/model"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/model"
    )
    
    return tokenizer, model

def get_current_metadata():
    return {
        "site_name": "YongDongIllHighSchool",
        "user_name": "홍길동",
        "user_role": "customer", # customer, admin
        "idu_name": "01_IB5",
        "idu_mapping": {
            "01_IB5": ["우리반"],
            "01_IB7": ["옆반"],
        },
        "modality_mapping": {
            "roomtemp": ["실내온도"],
            "settemp": ["설정온도"],
            "oper": ["전원"],
        },
        "current_datetime": "2022-09-30 12:00:00",
    }

def wait_for_input_from_user():
    return input("Enter your query: ")

def input_to_instruction_set(user_input, current_metadata):
    """
    Change text to instruction list.
    
    Parameters:
    - input_text (str): Input text to be changed to a instruction list.
    - current_metadata (dict): Not used yet.
    
    Returns:
    - semantic (dict): Semantic information of the input text.
    - instructions (list[Instruction]): List of instructions.
    
    Example:
        input: 오늘 우리반과 옆반의 온도차이 알려줘.
        return:
            semantic: Semantic(
                Temporal=[("오늘", "2022-09-30 00:00:00 ~ 2022-09-30 23:59:59")],
                Spatial=[('우리반', '01_IB5'), ('옆반', '01_IB7')]
                Modality=[("온도", 'roomtemp')],
                Type_Quantity=["diff"],
                Target=["온도"]
                Question_Actuation=["알려줘"]
            ), instructions: [
                Instruction(operation_flag="q", content="오늘 우리반과 옆반의 온도차이 알려줘", save_variable="V_1"),
                Instruction(operation_flag="r", using_varables="V_1", example="예 ) '우리반과 옆반의 온도차이는 2도입니다'")
            ]
    """
    semantic, instructions = InputToInstruction.execute(user_input)
    
    # Map spatial context
    for s_i, s in enumerate(semantic.Spatial):
        s_idu = "None"
        for idu, idu_repr in current_metadata['idu_mapping'].items():
            if s in idu_repr:
                s_idu = idu
                break
        semantic.Spatial[s_i] = (s, s_idu)
    
    # Map modality context
    for m_i, m in enumerate(semantic.Modality):
        m_col = "None"
        for column, col_repr in current_metadata['modality_mapping'].items():
            if m in col_repr:
                m_col = column
                break
        semantic.Modality[m_i] = (m, m_col)
    
    
    return semantic, instructions

def execute_query(semantic:Semantic, instruction:Instruction, execution_state:dict):
    """
    Execute query.
    """

    # Replace semantics in the instruction content to specific values.
    for semantics in [semantic.Temporal, semantic.Spatial, semantic.Modality]:
        for i, (k, v) in enumerate(semantics):
            instruction.content = instruction.content.replace(k, v)
    
    # If the instruction content contains "None", skip the execution.
    if "None" in instruction.content:
        execution_state['var'][instruction.save_variable] = None
        # logger.info(f"Instruction content contains 'None': {instruction.content}")
        return
    else:
        # Run the query generation LLM model
        sql_query = InstructionToSql.execute(instruction.content,str(semantic))
        
        # Execute query
        logger.info(f"sql_query: {sql_query}")
        sql_result = DBManager.execute_sql(sql_query)

        # Save the query result in execution state.
        execution_state['var'][instruction.save_variable] = sql_result
        logger.info(f"Saving {sql_result} to {instruction.save_variable}")
        return

def execute_response_generation(semantic:Semantic, instruction:Instruction, execution_state:dict, user_input:str, current_metadata:dict):
    """
    Generate a response based on the provided instruction.
    
    semantic: The semantic information of the input text.
    instruction: An Instruction object containing the response generation details.
    execution_state: A dictionary that stores the state during the execution, including variables.
    user_input: The user's input, which may influence the response.
    current_metadata: Additional metadata that may be relevant to response generation.
    
    Returns:
        - response (str): The generated response.
    """
    
    var = execution_state['var'][instruction.using_varables]
    if var is None:
        response = "죄송합니다. 해당 정보를 찾을 수 없습니다. (이유 설명 필요)"
    else:
        response = ResponseGeneration.execute(str(var), user_input, current_metadata)
    
    return response

def execute_instruction_set(semantic:Semantic, instruction_set:list[Instruction], user_input:str, current_metadata:dict, response_function:Callable):
    """
    Implement the agent to execute a set of instructions.
    
    semantic: The semantic information of the input text.
    instruction_set: A list of Instruction objects to be executed sequentially.
    user_input: The user's input, which may influence the execution flow.
    current_metadata: Additional metadata that may be relevant to the execution.
    """
    
    # This holds the state(e.x., variables) of the execution.
    execution_state = {
        "var": defaultdict(lambda: None) # This holds the variables generated during the execution.
    }
    
    logger.info(semantic)
    for instruction in instruction_set:
        logger.info(f"Executing instruction: {instruction}")
        
        if instruction.operation_flag == "q":
            # Execute query
            execute_query(semantic, instruction, execution_state)
            
        elif instruction.operation_flag == "r":
            # Execute response generation
            response = execute_response_generation(semantic, instruction, execution_state, user_input, current_metadata)
            response_function(response)
            