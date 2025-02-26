__all__ = [
    "load_models",
    "get_current_metadata",
    "wait_for_input_from_user",
    "input_to_instruction_set",
    "execute_query",
    "execute_response_generation",
    "execute_instruction_set",
    "execute_instruction_set_web",
]

import logging
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
    
    bnb_config = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=True,
        load_in_8bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir="/model"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir="/model"
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    return tokenizer, model, terminators
    

def load_sql_model():
    
    bnb_config = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=True,
        load_in_8bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
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
        cache_dir="/model",
        quantization_config=bnb_config
    )
    
    return tokenizer, model

def get_current_metadata() -> dict:
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
    - current_metadata (dict): Metadata that may be relevant to the input text.
    
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
            ), instructions: [
                    Instruction(operation_flag="q", content="오늘 우리반 온도 알려줘", save_variable="오늘 우리반 온도","값"),
                    Instruction(operation_flag="q", content="오늘 옆반 온도 알려줘", save_variable="오늘 옆반 온도","값"),
                    Instruction(operation_flag="r", example="예 ) '우리반과 옆반의 온도차이는 2도입니다'")
            ]
    """
    full_semantic, instructions = InputToInstruction.execute(user_input, current_metadata)
    # Map spatial and modality context for each instruction's semantic
    for instruction in instructions:
        if instruction.operation_flag == "r":
            continue  # Skip instructions with tag "r"
        # Map spatial context for the instruction's semantic
        for s_i, s in enumerate(instruction.semantic.Spatial):
            s_idu = "None"
            for idu, idu_repr in current_metadata['idu_mapping'].items():
                if s in idu_repr:
                    s_idu = idu
                    break
            instruction.semantic.Spatial[s_i] = (s, s_idu)

        # Map modality context for the instruction's semantic
        for m_i, m in enumerate(instruction.semantic.Modality):
            m_col = "None"
            for column, col_repr in current_metadata['modality_mapping'].items():
                if m in col_repr:
                    m_col = column
                    break
            instruction.semantic.Modality[m_i] = (m, m_col)

   
    logger.info(f"instructions: {instructions}")
    return full_semantic, instructions

def execute_query(instruction:Instruction):
    """
    Execute a SQL query based on the provided semantic information and instruction.

    Parameters:
    - semantic (Semantic): Contains semantic information that defines the context of the query.
    - instruction (Instruction): An object detailing the instruction for executing the query, including content to be transformed into SQL.
    
    Returns:
    - sql_query (str): The generated SQL query.
    """
   # Replace semantics in the instruction content to specific values.
    for semantics in [instruction.semantic.Temporal, instruction.semantic.Spatial, instruction.semantic.Modality]:
        for i, (k, v) in enumerate(semantics):
            instruction.content = instruction.content.replace(k, v)    
  
    
    # If the instruction content contains "None", skip the execution.
    
    if "None" in instruction.content:
        instruction.value = None
        # logger.info(f"Instruction content contains 'None': {instruction.content}")
        return
    else:
        # Run the query generation LLM model
        sql_query = InstructionToSql.execute(instruction, str(instruction.semantic))
        
        # Execute query
        sql_result = DBManager.execute_sql(sql_query)
        if len(sql_result) > 0:
            sql_result = sql_result[0]
        logger.info(f"sql_query: {sql_query}, sql_result: {sql_result}")

        # Save the query result in execution state.
        instruction.value = sql_result
        return sql_query
 

def execute_response_generation(instruction:Instruction, query_mapping : list[list[str, str]], user_input:str, current_metadata:dict):
    """
    Generate a response based on the provided instruction.
    
    instruction: An Instruction object containing the response generation details.
    query_mapping (list[list[str, str]]): A mapping that associates SQL query results with their corresponding variable names.
    user_input: The user's input, which may influence the response.
    current_metadata: Additional metadata that may be relevant to response generation.
    
    Returns:
        - response (str): The generated response.
    """
    for k,v in query_mapping :
        if v is None or len(v)==0 :
            response = "죄송합니다. 해당 정보를 찾을 수 없습니다. (이유 설명 필요)"
        else:
            response = ResponseGeneration.execute(instruction, query_mapping, user_input, current_metadata)
        
    return response

def execute_instruction_set(semantic:Semantic,instruction_set:list[Instruction], user_input:str, current_metadata:dict, response_function:Callable):
    """
    Implement the agent to execute a set of instructions.
    
    semantic: The semantic information of the input text.
    instruction_set: A list of Instruction objects to be executed sequentially.
    user_input: The user's input, which may influence the execution flow.
    current_metadata: Additional metadata that may be relevant to the execution.
    """
    query_mapping = []
    for instruction in instruction_set:
        logger.info(f"Executing instruction: {instruction}")
        
        if instruction.operation_flag == "q":
            # Execute query
            execute_query(instruction)
            query_mapping.append([instruction.save_variable,instruction.value])
            
        elif instruction.operation_flag == "r":
            pass
            # Execute response generation
            response = execute_response_generation(instruction, query_mapping, user_input, current_metadata)
            response_function(response)
            
def execute_instruction_set_web(semantic:Semantic, instruction_set:list[Instruction], user_input:str, current_metadata:dict, response_function:Callable):
    """
    This function is a duplicate of execute_instruction_set, but with additional support for web-based responses.
    """
    query_mapping = []
    for instruction in instruction_set:
        logger.info(f"Executing instruction: {instruction}")
        
        yield from response_function(f"<h1 Executing instruction: {instruction}>")
        if instruction.operation_flag == "q":
            # Execute query
            generated_sql = execute_query(instruction)
            query_mapping.append([instruction.save_variable,instruction.value])
            generated_sql = generated_sql.replace("\n", "").replace("         ", "")
            yield from response_function("<h2 Generated SQL:>")
            yield from response_function(generated_sql)
            yield from response_function(f"<h2 Query result:>")
            yield from response_function(str(instruction.value))
        
        elif instruction.operation_flag == "r":
            # Execute response generation
            response = execute_response_generation(instruction, query_mapping, user_input, current_metadata)
            yield from response_function("<h2 Generated response:>")
            yield from response_function(f"<h3 {response}>")