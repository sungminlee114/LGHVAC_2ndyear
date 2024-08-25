import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ResponseGeneration:
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
    
 
    
    @classmethod
    def execute(cls, sql_result: str, user_input: str, current_metadata: dict) -> str:
        """
        Generate a response based on the SQL result and user input.

        Parameters:
        - sql_result (str): The result from a SQL query to be included in the response.
        - user_input (str): The user's question that needs to be addressed.
        - current_metadata (dict): Additional metadata about the current context (not used in this method).

        Returns:
        - str: A generated response combining the SQL result and user input in a polite and natural manner.
        """
        
        new_str = "질문인 "+user_input + "와 답변인 " + sql_result + "를 합쳐서 한줄의 친절한 답변 문장으로 써줘."
        PROMPT = '''
        친절한 답변 해주세요. 답변은 "-입니다" 로 끝나게 적성해줘. 조사가 문법에 맞게 작성되게 해줘. 전력 사용량의 단위는 Wh 야. 
        답변 문장의 형태는 다음과 같아. {user_input} 은 {sql_result} 입니다. 
        예시)
        1. user_input: 오늘 우리집 3시의 전력사용량 알려줘
          sql_result: 30
          답변: 오늘 우리집 3시의 전력사용량은 30Wh입니다.

        '''
        messages = [
            {"role": "system", "content": f"{PROMPT}"},
            {"role": "user", "content": f"{new_str}"}
        ]

        input_ids = cls.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(cls.model.device)

        outputs = cls.model.generate(
            input_ids,
            max_new_tokens=2048,
            eos_token_id=cls.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty = 1.1
        )

        result_string:str = cls.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        return result_string

if __name__ == "__main__":
    result = ResponseGeneration.execute(
       "5", "오늘 우리집과 옆집의 실내 온도 차이 알려줘", {
        "site_name": "YongDongIllHighSchool",
        "user_name": "홍길동",
        "user_role": "customer",  # customer, admin
        "idu_name": "01_IB5",
        "current_datetime": "2022-09-30 12:00:00",
    })
    
    print(result)