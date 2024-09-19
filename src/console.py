import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.core import *  

if __name__ == "__main__":
    load_models()
    while True:
        try:
            user_input = wait_for_input_from_user()
            current_metadata = get_current_metadata()
            
            semantic, instruction_set = input_to_instruction_set(user_input, current_metadata)
            
            execute_instruction_set(semantic, instruction_set, user_input, current_metadata, response_function=lambda r: print(f"답변: {r}"))
        except Exception as e:
            import traceback
            traceback.print_exception(e)