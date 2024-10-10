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
    원본 질문, query_mapping, current_metadata 가 주어졌을 때, 종합해서 친절한 답변을 작성해줘.
    query_mapping은 [["변수", "값"], ...]으로 이루어져있어. 예) [["현재 실내 온도", 3],]
    query_mapping 은 user_input 에 대한 해답을 하기 위해 필요한 데이터의 값이 들어있는 list 야. query_mapping 의 각 원소 중 두 번째 원소가 답변에 해당하니 이를 활용해줘.  
    user_input 에서 답변을 하기 위해 필요한 데이터 값을 query_mapping 에 주어진 [["변수","값"]...] 활용해서 답변을 해줘. 각 변수의 값이 뭔지를 정확하게 하고, 묻는 것이 무엇인가를 정확하게 확인하고 오류없이 답변해줘.
    답변 추론 과정은 다음과 같아.
    1. 질문 분석: user_input을 통해 사용자가 요청하는 정보의 종류를 파악합니다.
    2. 키워드 확인: query_mapping에서 관련된 키워드를 식별하여 필요한 데이터를 정의합니다.
    3. 정보 수집: query_mapping을 활용해 해당 데이터(예: 실내온도)를 추출합니다.
    4. 비교 및 계산: 필요한 경우, 수집한 정보를 기반으로 비교 또는 계산을 수행합니다.
    5. 결과 도출: current_metadata를 통해 사용자의 상황을 고려하여 최종 정보를 정리합니다.
    6. 답변 제공: 사용자의 질문에 맞춰 명확한 형태로 결과를 전달합니다.
    
    Example)
        user_input :"지금 가장 더운 반 알려줘"
            1. 질문 분석: 사용자가 요청하는 정보의 종류를 파악합니다. "가장 더운 반"이라는 표현에서 '더운'은 온도와 관련이 있음을 이해합니다.
            2. 키워드 확인:
            "반"은 특정 공간(우리반, 옆반)을 나타내고,
            "더운"은 온도를 기준으로 비교할 것을 암시합니다.
            3. 정보 수집: 이전 단계에서 수집한 데이터를 활용하여 각 반의 실내온도를 가져옵니다.
            예: 우리반의 온도(28.0도), 옆반의 온도(27.0도).
            4. 비교 및 판단:
            수집한 온도를 비교하여 가장 높은 온도를 가진 반을 식별합니다.
            예: 28.0도(우리반) > 27.0도(옆반).
            5. 결과 도출: "우리반"이 가장 높은 온도를 기록했음을 정리합니다.
            6. 답변 제공: user_input 에 맞춰 "지금 가장 더운 반은 우리반입니다."라고 명확하게 전달합니다.

    답변을 할때, 꼭 current_metadata를 확인하고 제대로 된 변수에 제대로 된 값을 말하고 있는건지 꼭 확인해줘.
    만약 질문에 알맞은 값이 없다면, 답변 예시를 그대로 사용하지 말고 답변할 수 없다고 해줘.
    대답을 추론하는 과정은 말할 필요 없어. 그냥 결론만 말해줘.
    추론 과정은 답변하지 않지만 앞에 질몬이 뭐였는지는 요약해줘.
    답변은 한번만 해줘.
    답변에 "assistant" 가 안나오도록 해줘. 
    query_mapping 에서 mapping 된 값이 None 이면 값이 "값이 없습니다" 라고 대답해줘.
    
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
            temperature=0.001,
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