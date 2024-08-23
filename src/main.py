import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from collections import defaultdict

from src.input_to_instructions.load_and_execute import InputToInstruction, Instruction
from src.db.manager import DBManager

def get_current_metadata():
    return {
        "site_name": "YongDongIllHighSchool",
        "user_name": "홍길동",
        "user_role": "customer", # customer, admin
        "idu_name": "01_IB5",
        "current_datetime": "2022-09-30 12:00:00",
    }

def wait_for_input_from_user():
    return input("Enter your query: ")

def input_to_instruction_set(user_input, current_metadata) -> list[Instruction]:
    """
    Change text to instruction list.
    
    Parameters:
    - input_text (str): Input text to be changed to a instruction list.
    - current_metadata (dict): Not used yet.
    
    Returns:
    - instructions (list[Instruction]): List of instructions.
    
    Example:
        input: "지난 여름 우리집과 옆집의 전력사용량 비교해줘"
        output: [
            Instruction(operation_flag="q", content="지난 여름 우리집 전력사용량 알려줘"),
            Instruction(operation_flag="q", content="지난 여름 옆집 전력사용량 알려줘"),
            Instruction(operation_flag="o", content="final_result = a_1 - a_2"),
            Instruction(operation_flag="r", content="final_result를 한국어로 답해줘")
        ]
    """
    return InputToInstruction.execute(user_input)

def execute_query(instruction:Instruction, execution_state:dict):
    """
    Execute query.
    @민주
    """
    
    # Run the query generation LLM model
    
    # Execute query
    sql_result = DBManager.execute_sql(sql_string)
    
    # Save the query result in execution state.
    current_variable_name = f"a_{execution_state['op_counts'][instruction.operation_flag]}" # a_1, a_2, 3_a, ... 변수 이름 생성
    execution_state['var'][current_variable_name] = sql_result
    
    return

def execute_operation(instruction:Instruction, execution_state:dict):
    """
    Execute the operation based on the provided instruction.
    
    instruction: An Instruction object containing the operation to be executed.
    execution_state: A dictionary that stores the state during the execution, including variables.
    """
    
    # TODO: Can be optimized.
    # Loop through the variables and execute the operation to store the variable in memory.
    for key, value in execution_state['var'].items():
        exec(f"{key} = {value}")
    
    expression = instruction.content # Exmple, "final_result = a_1 - a_2"
    exec(expression)
    variable_name = expression.split("=")[0].strip() # "final_result"
    execution_state['var'][variable_name] = eval(variable_name) # exec("final_result") = final_result
    
    return

def execute_response_generation(instruction:Instruction, execution_state:dict, user_input:str, current_metadata:dict):
    """
    Generate a response based on the provided instruction.
    
    instruction: An Instruction object containing the response generation details.
    execution_state: A dictionary that stores the state during the execution, including variables.
    user_input: The user's input, which may influence the response.
    current_metadata: Additional metadata that may be relevant to response generation.
    """
    
    # Assume the response is using only 'final_result' variable.
    response = instruction.content # Exmple, "final_result를 한국어로 답해줘"
    response = response.replace("final_result", str(execution_state['var']['final_result']))
    
    # Run response generation LLM model. @민주
    
    print(f"답변: {response}")

def execute_instruction_set(instruction_set:list[Instruction], user_input:str, current_metadata:dict):
    """
    Implement the agent to execute a set of instructions.
    
    instruction_set: A list of Instruction objects to be executed sequentially.
    user_input: The user's input, which may influence the execution flow.
    current_metadata: Additional metadata that may be relevant to the execution.
    """
    
    # This holds the state(e.x., variables) of the execution.
    execution_state = {
        "op_counts": defaultdict(lambda: 0), # This holds the count of operations executed.
        "var": defaultdict(lambda: None) # This holds the variables generated during the execution.
    }
    
    for instruction in instruction_set:
        logger.info(f"Executing instruction: {instruction}")
        
        execution_state["op_counts"][instruction.operation_flag] += 1
        
        if instruction.operation_flag == "q":
            # Execute query
            execute_query(instruction, execution_state)
            
        elif instruction.operation_flag == "o":
            # Execute operation
            execute_operation(instruction, execution_state)
        elif instruction.operation_flag == "r":
            # Execute response generation
            execute_response_generation(instruction, execution_state, user_input, current_metadata)

def main():
    while True:
        user_input = wait_for_input_from_user()
        current_metadata = get_current_metadata()
        
        instruction_set = input_to_instruction_set(user_input, current_metadata)
        
        execute_instruction_set(instruction_set, user_input, current_metadata)


if __name__ == "__main__":
    main()