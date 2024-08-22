import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class InputToInstruction:
    model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    PROMPT = "세부문장은 리스트에 들어가야하는데, 예를 들어 오늘 우리집과 옆집의 온도차이 알려줘 라는 문장은 total_list = [[q,'오늘 우리집 온도 알려줘',a],[q,'오늘 옆집 온도 알려줘',a],[r,total_list[0][-1]-total_list[1][-1]]] 야. 마지막 result 까지 리스트안에 넣어서 해줘. 특별한 문장 간 연산이 필요없는 경우에는, 예를 들어 지금 현재 온도 알려줘 라는 문장은 total_list = [q,'지금 현재온도 알려줘',a] , 올해 에어컨 사용료가 가장 많았던 달이 언제야 ? total_list=[q,'올해 에어컨 사용료가 가장 많았던 달은 언제야?,'a'']"
    
    @classmethod
    def parse_from_keyword(cls, gpt_results, keyword):
        # keyword가 total_list에 있는지 확인하고, 위치를 찾습니다.
        start_index = gpt_results.find(keyword)
        
        # keyword가 존재하면, 그 위치부터 끝까지의 문자열을 반환합니다.
        if start_index != -1:
            return gpt_results[start_index:]
        else:
            # keyword가 존재하지 않으면 None을 반환합니다.
            return None

    @classmethod
    def execute(cls, instruction):
        """
        ~~
        
        Parameters:
        - instruction (str): ~
        
        Returns:
        - ~
        
        """
        
        
        
        messages = [
        {"role": "system", "content": f"{cls.PROMPT}"},
        {"role": "user", "content": f"{instruction}"}
        ]

        input_ids = cls.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(cls.model.device)

        outputs = cls.model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=cls.terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty = 1.1
            )

        result_string = cls.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        parsed_string:str = cls.parse_from_keyword(result_string, 'total_list')
        return parsed_string
