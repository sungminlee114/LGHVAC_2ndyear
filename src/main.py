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
    Implement core functions under input_to_instruction_set module and call it here.
    
    Document the output format of the function here.
    """

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