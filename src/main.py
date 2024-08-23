from src.db.manager import DBManager

def get_current_metadata():
    return {
        "site_name": "YongDongIllHighSchool",
        "user_name": "홍길동",
        "user_role": "customer", # customer, admin
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
    - instructions (list[Instruction]): List of instructions.
    
    Example:
        input: "지난 여름 우리집과 옆집의 전력사용량 비교해줘"
        output: [
            Instruction(operation_flag="q", content="지난 여름 우리집 전력사용량 알려줘"),
            Instruction(operation_flag="q", content="지난 여름 옆집 전력사용량 알려줘"),
            Instruction(operation_flag="o", content="final_result = 1_a - 2_a"),
            Instruction(operation_flag="r", content="final_result를 한국어로 답해줘")
        ]
    """
    return InputToInstruction.execute(user_input)

def execute_instruction_set(instruction_set):
    """
    Implement agent.
    """
    
    for instruction in instruction_set:
        pass

def main():
    while True:
        user_input = wait_for_input_from_user()
        current_metadata = get_current_metadata()
        
        instruction_set = input_to_instruction_set(user_input, current_metadata)
        
        execute_instruction_set(instruction_set)


if __name__ == "__main__":
    main()