import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.input_to_instructions.load_and_execute import Instruction

class ResponseGeneration:
    PROMPT = '''
    친절한 답변 해줘. 조사가 문법에 맞게 작성되게 해줘.
    원본 질문, 답변에 사용될 값, sql_query, sql_result, query_mapping, 답변 예시가 주어졌을 때, 종합해서 친절한 답변을 작성해줘.
    query_mapping은 [["질문", "답변"], ...]으로 이루어져있어. 예) [["현재 실내 온도", 3],]
    query_mapping 은 user_input 에 대한 해답을 하기 위해 필요한 데이터의 값이 들어있는 list 야. query_mapping 의 각 원소 중 두 번째 원소가 답변에 해당하니 이를 활용해줘.  
    user_input 에서 답변을 하기 위해 필요한 데이터 값을 query_mapping 에 주어진 [["질문","답변"]...] 활용해서 원하는 답변을 해줘. 
    답변을 할때, 꼭 current_metadata를 확인하고 제대로 된 변수에 제대로 된 값을 말하고 있는건지 꼭 확인해줘. 이때 구체적인 질문인 specific_query 를 참고하여 활용해줘.
    만약에 sql_result 가 idu.name 이나 idu.id 으로 되어있으면 current_metadata 의 idu_mapping 의 값으로 바꿔줘.
    sql_result의 값들을 current_metadata를 사용해서 사람이 알아듣기 좋게 바꿔줘.
    만약 질문에 알맞은 값이 없다면, 답변 예시를 그대로 사용하지 말고 답변할 수 없다고 해줘.
    대답을 추론하는 과정은 말할 필요 없어. 그냥 결론만 말해줘.
    추론 과정은 답변하지 않지만 앞에 질몬이 뭐였는지는 요약해줘.
    답변은 한번만 해줘.
    sql query 다음에 "assistant" 안나오도록 해줘. 
    '''
 
    
    @classmethod
    def execute(cls,instruction:Instruction, query_mapping, user_input: str, current_metadata: dict) -> str:
        """
        Generate a response based on the SQL result and user input.

        Parameters:
        - instruction (Instruction): An object containing details for generating the response, including an example response format.
        - query_mapping (list[list[str, str]]): A mapping that links specific questions to their corresponding answers derived from SQL results.
        - user_input (str): The user's question that needs to be addressed in the response.
        - current_metadata (dict): Additional metadata about the current context, including mappings for user-friendly output (e.g., `idu_mapping`).

        Returns:
        - str: A generated response combining the SQL result and user input in a polite and natural manner.
        """
        
        formatted_metadata = formatted_metadata = "\n".join([f"\t\t{key}: {value}" for key, value in current_metadata.items()])
        
        INPUT = f'''
        원본 질문: {user_input}
        query_mapping: {query_mapping}
        current_metadata: \n{formatted_metadata}
        답변 예시: {instruction.example}
        '''
        
        logger.info(f"Response generator input: {INPUT}")
        
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