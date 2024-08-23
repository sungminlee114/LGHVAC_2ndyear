import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Instruction:
    """
    operation_flag: choice of [
        "q" (question),
        "o" (operation),
        "r" (response)
    ]
    
    content: str
    """
    def __init__(self, instruction):
        self.operation_flag:str = instruction[0]
        self.content:str = instruction[1]
    
    def __repr__(self) -> str:
        return f"Instruction(operation_flag={self.operation_flag}, content={self.content})"
class InputToInstruction:
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
    
    PROMPT = """세부문장은 리스트에 들어가야하는데, 
세부 문장의 예시는 다음과 같아. 세부 문장은 나중에 데이터 베이스를 query하는데 쓰일 문장이고, 세부 문장을 나누는 기준은 하나의 테이블(장소) 로 해결될 수 있는지 없는지야.오늘 우리집과 옆집의 온도차이 알려줘 라는 문은 여러 테이블(장소)에 접근해야해. 
따라서 세부 문장은 첫번째, 오늘 우리집의 온도를 알려줘. 두번째, 옆집의 온도를 알려줘. 그리고 결과로는 첫번째 쿼리의 답과 두번째 쿼리의 답을 뺀거야. 최종 r부분에 답을 할때 첫번째 문장이름은 1_q, 첫번째 문장의 답 이름은 1_a이야. 그리고 찻반쩨 문장의 장소 이름은 1_site 로 해줘. 최종 답은 딱 다음과 같은 리스트의 형태로 똑같이 나와야해. [[q,오늘 우리집 온도 알려줘],[q,오늘 옆집 온도 알려줘],[o,final_result =1_a-2_a],[r, "final_result를 한국어로 답해줘"]] 야. 두번째 예시로는 지난 여름 우리집 평균온도 알려줘 라는 문장은 장소를 데이터베이스 테이블 이름이라고 할때  하나의 테이블(장소)에서 해결할 수 있는 세부문장이 없는 문장이야. 따라서 답은 다음 리스트이 형태 그대로 똑같이 나와야해. 변수 바꾸지마. [[q,지난 여름 우리집 평균온도 알려줘],[o,final_result=1_a],[r, "final_result를 한국어로 답해줘"]] 야.
예시를 하나 더 들면, 이번달 중 우리집 온도가 가장 추운달이 언제야 라는 문장은 하나의 테이블(장소)로 해결될수 있으니 답은 다음과 같이 똑같이 나와야해. 변수 바꾸지마. [q, 이번달 중 우리집 온도가 가장 추운달이 언제야][o,final_result=1_a],[r, "final_result를 한국어로 답해줘"]]. 예시를 하나 더 들면, 우리집과 오피스텔 101동중에 최고온도가 더 낮은 곳은 어디야? 의 세부문장 리스트 알려줘 라는 문장은
[
    [q, 우리집의 최고 온도가 얼마야?],
    [q, 오피스텔 101동의 최고 온도가 얼마야?],
    [o, final_result = "1_site" if 1_a < 2_a else "2_site"],
    [r, "final_result를 한국어로 답해줘"]
]

세부 문장을 나누는 기준은 장소야. 
우리집 실내온도만 물으면 비교할 필요없어. 장소가 여러개 나올때만 여러 세부문장으로 나누게 해줘.
변수 바꾸지말고 출력해줘.

"""

    @classmethod
    def execute(cls, input_text:str) -> list[Instruction]:
        """
        Change text to instruction list.
        
        Parameters:
        - input_text (str): Input text to be changed to a instruction list.
        
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
        
        messages = [
            {"role": "system", "content": f"{cls.PROMPT}"},
            {"role": "user", "content": f"{input_text}"}
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

        result_string:str = cls.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        """
        "[[q, 지난 여름 우리집 전력사용량 알려줘],
        [q, 지난 여름 옆집 전력사용량 알려줘],
        [o, final_result = 1_a - 2_a],
        [r, "final_result를 한국어로 답해줘"]]"
         
        이런 형식의 result_string(str)을 실제 list(list[str])로 변환하는 코드를 작성해주세요.
        """
        instructions = []
        result_string = result_string.replace("[[", "").replace("]]", "").replace("\n", "").replace("\t", "").replace('"', "")
        result_string = result_string.split("],")
        for instruction in result_string:
            instruction = instruction.replace("[", "").replace("]", "")
            instruction = instruction.split(", ")
            instructions.append(Instruction(instruction))
        
        return instructions

if __name__ == "__main__":
    result = InputToInstruction.execute(
        "지난 여름 우리집과 옆집의 전력사용량 비교해줘"
    )
    
    print(result)