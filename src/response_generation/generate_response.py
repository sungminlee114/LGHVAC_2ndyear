import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.input_to_instructions.load_and_execute import Instruction

class ResponseGeneration:
    PROMPT = '''
    친절한 답변 해줘. 답변은 "-입니다" 로 끝나게 적성해줘. 조사가 문법에 맞게 작성되게 해줘.
    원본 질문, 답변에 사용될 값, 답변 예시가 주어졌을 때, 질문과 답변을 합쳐서 친절한 답변을 작성해줘.
    '''
 
    
    @classmethod
    def execute(cls, sql_result: str, instruction:Instruction, user_input: str, current_metadata: dict) -> str:
        """
        Generate a response based on the SQL result and user input.

        Parameters:
        - sql_result (str): The result from a SQL query to be included in the response.
        - user_input (str): The user's question that needs to be addressed.
        - current_metadata (dict): Additional metadata about the current context (not used in this method).

        Returns:
        - str: A generated response combining the SQL result and user input in a polite and natural manner.
        """
        
        
        INPUT = f'''
        원본 질문: {user_input}
        답변에 사용될 값: {sql_result}
        답변 예시: {instruction.example}
        '''
        
        messages = [
            {"role": "system", "content": f"{cls.PROMPT}"},
            {"role": "user", "content": f"{INPUT}"}
        ]

        input_ids = cls.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(cls.model.device)

        outputs = cls.model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=cls.terminators,
            pad_token_id=cls.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty = 1.1
        )

        result_string:str = cls.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        return result_string

if __name__ == "__main__":
    from src.main import load_text_model
    
    tokenizer, model, terminators = load_text_model()
    ResponseGeneration.model = model
    ResponseGeneration.tokenizer = tokenizer
    ResponseGeneration.terminators = terminators
    
    
    result = ResponseGeneration.execute(
       "5", "오늘 우리집과 옆집의 실내 온도 차이 알려줘", {
        "site_name": "YongDongIllHighSchool",
        "user_name": "홍길동",
        "user_role": "customer",  # customer, admin
        "idu_name": "01_IB5",
        "current_datetime": "2022-09-30 12:00:00",
    })
    
    print(result)