import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Instruction:
    """
    operation_flag: choice of [
        "q" (question),
        "r" (response)
    ]
    
    -- if operation_flag is "q" --
    content: str
    save_variable: str
    
    -- if operation_flag is "r" --
    using_varables: str
    example: str
    """
    def __init__(self, instruction):
        self.operation_flag:str = instruction[0]
        assert self.operation_flag in ["q", "r"], instruction
        
        if self.operation_flag in ["q"]:
            self.content:str = str(instruction[1])
            self.save_variable:str = str(instruction[2])
        elif self.operation_flag == "r":
            self.using_varables:str = str(instruction[1])
            self.example:str = str(instruction[2])
    
    def __repr__(self) -> str:
        if self.operation_flag == "r":
            return f"Instruction({[self.operation_flag, self.using_varables, self.example]})"
        else:
            return f"Instruction({[self.operation_flag, self.content, self.save_variable]})"

class Semantic:
    """
    Temporal: list of tuple
    Spatial: list of tuple
    Modality: list
    Type_Quantity: list
    Target: list
    Question_Actuation: list
    """
    
    def __init__(self, semantic):
        self.Temporal = semantic["Temporal"]
        self.Spatial = semantic["Spatial"]
        self.Modality = semantic["Modality"]
        self.Type_Quantity = semantic["Type/Quantity"]
        self.Target = semantic["Target"]
        self.Question_Actuation = semantic["Question/Actuation"]

    def __repr__(self) -> str:
        return f"Semantic(Temporal={self.Temporal}, Spatial={self.Spatial}, Modality={self.Modality}, Type_Quantity={self.Type_Quantity}, Target={self.Target}, Question_Actuation={self.Question_Actuation})"
    
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
    
    PROMPT = \
"""너는 훌룡한 AI HVAC 챗봇이야. 거짓 정보들은 말하지 말아줘.
사용자의 입력에 대해 다음을 고려해 출력해줘.
출력은 무조건 서론 없이 dictionary 형태로 해줘. 다른 형태는 용납불가능해.
"이 질문에 대한 답변은 다음과 같습니다"와 같은 문장은 절대로 사용하지 말아줘.

<사전정보>
위치: '영동일고등학교'.
질문자: 이름:'홍길동', 분류: '손님'
Modality: ['실내온도', '설정온도', '전원']
현재시간: '2022-09-30 12:00:00'

<출력 형태 및 조건>
출력은 1. '질문 요약 reasoning', 2. '질문 요약', 3. 'Semantic Parsing reasonsing', 4. 'Semantic Parsing'(dict), 5. 'Instruction Set'(list)
1. '질문 요약 reasoning': 사용자의 질문을 요약하는 과정에서 이유를 설명. 쓰여있는 정보를 활용하고, 없으면 추측한다. Semantic parsing보다 추상적이어야 함.
2. '질문 요약': 사용자의 질문을 요약한 내용.
3. 'Semantic Parsing reasonsing': Semantic Parsing을 하는 과정에서 이유를 설명. 이 과정에서 사전정보에 없는 장소나 구체적인 시간/시간기간을 추측해야만 함.
4. 'Semantic Parsing': Semantic Parsing 결과. dict 형태로 구성되며, Temporal, Spatial, Modality, Type/Quantity, Target, Question/Actuation key를 가짐.
5. 'Instruction Set': Instruction Set 결과. list 형태로 구성되며, 각 원소는 [flag, content, variable]로 구성됨.
출력은 무조건 예시의 dictionary 형태를 지켜줘. 무조건. 제일 중요해.

<Semantic Parsing 조건>
Temporal: 시간 혹은 시간 범위를 나타내는 정보. 사전정보 참고. 대표적 representation과 Timestamp 형식의 튜플로 표현됨. 아무런 정보가 없을 경우 '지금'으로 표현되지만, 정보가 있는 경우 자제해야 함.
Spatial: 공간 혹은 공간 범위를 나타내는 정보. 사전정보 참고. 대표적 representation으로 표현됨. 아무런 정보가 없을 경우 '우리반'으로 표현되지만, 정보가 있는 경우 자제해야 함.
Modality: 정보의 형태 혹은 정보의 속성. 사전정보를 참고해서 제일 유사한 단어로 표현해야 하지만, 사전정보에 없는 경우 추정해야 함.
Type/Quantity: 정보의 종류 혹은 양. 
Target: 정보의 대상.
Question/Actuation: 질문 혹은 행위.

<Instruction Set 조건>
공통: 'Semantic Parsing'의 정보를 활용하여 질문을 만들어야 함. 알 수 없음이 포함될 수 없다.
q: Query을 나타내는 flag. 두 번째 인자는 Query Generator에서 사용할 예정이며, 세 번째 인자는 query의 결과를 저장할 변수명이다.
r: Response를 나타내는 flag. 두 번째 인자는 Response를 제작하는 데 사용할 변수명이며, 세 번째 인자는 Response의 예시를 나타낸다.

<절대로 하지 말아야 할 것>
출력 앞 뒤에 불필요한 내용이 있거나 dictionary 형태를 지키지 않으면 절대로 안됨(즉, 중괄호가 무조건 있어야함). 특히 '~~질문에 대한 답변은 다음과 같습니다.나 주어진예시를~~'와 같은 문장은 절대로 작성하지 않아야 함.
이미 명시적으로 시간/공간 정보가 input에 있는데, 없다고 잘못 판단하면 안됨.

<예시들과 다른 입력이 들어올 경우>
예시들과 조금 다르더라도 출력의 형식은 무조건 지켜야함.

<좋은 예시1>
입력: 오늘 우리반과 옆반의 온도차이 알려줘.
출력:
"{
    "질문 요약 reasoning": "시간은 오늘이라고 쓰여 있다. 장소는 우리반과 옆반이라고 쓰여 있다. 온도는 실내온도로 추정된다.",
    "질문 요약": "오늘 우리반과 옆반의 실내온도 차이 알려줘.",
    "Semantic Parsing reasonsing": "오늘은 2022년 9월 30일이다."
    "Semantic Parsing": {
        "Temporal": [("오늘", "2022-09-30 00:00:00 ~ 2022-09-30 23:59:59")],
        "Spatial": ["우리반", "옆반"],
        "Modality": ["실내온도"],
        "Type/Quantity": ["diff"],
        "Target": ["온도"],
        "Question/Actuation": ["알려줘"]
    },
    "Instruction Set": [
        ["q", "오늘 우리반과 옆반의 실내온도 차이 알려줘.", "V_1"],
        ["r", "V_1", "예) '우리반이 옆반보다 2도 높아요'"]
    ]
}"

<좋은 예시2>
입력: 지난 여름 우리반 평균 설정온도 알려줘.
출력: 
"{
    "질문 요약 reasoning": "시간은 지난 여름라고 쓰여 있다. 지난 여름은 올해 여름이라는 뜻으로 추정된다. 장소는 우리반이라고 쓰여있다.",
    "질문 요약": "올해 여름 우리반 평균 설정온도 알려줘.",
    "Semantic Parsing reasonsing": "올해는 2022년이며, 여름은 6월 1일부터 8월 31일까지이다."
    "Semantic Parsing": {
        "Temporal": [("올해 여름", "2022-06-01 00:00:00 ~ 2022-08-31 23:59:59")],
        "Spatial": ["우리반"],
        "Modality": ["설정온도"],
        "Type/Quantity": ["mean"],
        "Target": ["온도"],
        "Question/Actuation": ["알려줘"]
    },
    "Instruction Set": [
        ["q", "올해 여름 우리반 평균 설정온도 알려줘.", "V_1"],
        ["r", "V_1", "예) '지난 여름 우리반의 평균 설정온도는 26도입니다.'"]
    ],
}"

<좋은 예시3>
입력: 이번달 중 우리반 온도가 가장 더운날이 언제야?
출력:
"{
    "질문 요약 reasoning": "시간은 이번달이라고 쓰여있다. 장소는 우리반이라고 쓰여 있다. 온도는 실내온도로 추정된다. 더운날은 온도가 가장 높은 날을 의미한다."
    "질문 요약": "이번달 우리반 실내온도가 가장 높은 날짜 알려줘."
    "Semantic Parsing reasonsing": "이번달은 9월 1일부터 9월 30일이다. 가장 높은 온도가 아닌 그 날짜를 알아야 한다."
    "Semantic Parsing": {
        "Temporal": [("이번달", "2022-09-01 00:00:00 ~ 2022-09-30 23:59:59")],
        "Spatial": ["우리반"],
        "Modality": ["실내온도"],
        "Type/Quantity": ["argmax"],
        "Target": ["날짜"],
        "Question/Actuation": ["알려줘"]
    },
    "Instruction Set": [
        ["q", "이번달 우리반 실내온도가 가장 높은 날짜 알려줘.", "V_1"],
        ["r", "V_1", "예) '이번달 중 우리반의 온도가 가장 더운날은 2022년 9월 15일입니다.'"]
    ]
}"

<좋은 예시4>
입력: 우리반과옆반중에온도가더낮은곳은어디야?
출력: 
"{
    "질문 요약 reasoning": "시간은 지금으로 추측한다. 장소는 우리반과 옆반이라고 쓰여있다. 온도는 실내온도로 가정한다. 곳은 장소를 의미한다."
    "질문 요약": "지금 우리반과 옆반 중 실내온도가 더 낮은 장소 알려줘."
    "Semantic Parsing reasonsing": "지금은 2022년 9월 30일 12시 00분이다."
    "Semantic Parsing": {
        "Temporal": [("지금", "2022-09-30 12:00:00")],
        "Spatial": ["우리반", "옆반"],
        "Modality": ["실내온도"],
        "Type/Quantity": ["argmin"],
        "Target": ["장소"],
        "Question/Actuation": ["알려줘"]
    },
    "Instruction Set": [
        ["q", "지금 우리반과 옆반 중 실내온도가 더 낮은 장소 알려줘.", "V_1"],
        ["r", "V_1", "예) '우리반의 온도가 옆반보다 더 낮아요.'"]
    ]
}"

<좋은 예시5>
입력: 에어컨 켜져있어?
출력: 
"{
    "질문 요약 reasoning": "시간은 지금으로 추측한다. 장소는 우리반으로 추측한다. 에어컨의 전원 여부를 알려달라는 의미이다."
    "질문 요약": "지금 우리반 전원 값 알려줘."
    "Semantic Parsing reasonsing": "지금은 2022년 9월 30일 12시 00분이다."
    "Semantic Parsing": {
        "Temporal": [("지금", "2022-09-30 12:00:00")],
        "Spatial": ["우리반"],
        "Modality": ["전원"],
        "Type/Quantity": ["value"],
        "Target": ["전원"],
        "Question/Actuation": ["알려줘"]
    },
    "Instruction Set": [
        ["q", "지금 우리반 전원 값 알려줘.", "V_1"],
        ["r", "V_1", "예) '지금 우리반 에어컨은 켜져 있습니다.'"]
    ]
}"

<좋은 예시6>
입력: 앞반 지금 전원 켜져있어?
출력: 
"{
    "질문 요약 reasoning": "시간은 지금으로 추측한다. 장소는 앞반으로 쓰여있다. 전원에 대한 정보를 알고 싶어하는 것으로 추정된다."
    "질문 요약": "지금 앞반 전원 값 알려줘."
    "Semantic Parsing reasonsing": "지금은 2022년 9월 30일 12시 00분이다."
    "Semantic Parsing": {
        "Temporal": [("지금", "2022-09-30 12:00:00")],
        "Spatial": ["앞반"],
        "Modality": ["전원"],
        "Type/Quantity": ["value"],
        "Target": ["전원"],
        "Question/Actuation": ["알려줘"]
    },
    "Instruction Set": [
        ["q", "지금 앞반 전원 값 알려줘.", "V_1"],
        ["r", "V_1", "예) '지금 앞반 에어컨은 꺼져 있습니다.'"]
    ]
}"

<좋은 예시7>
입력: 우리집 지금 전력사용량 얼마야?
출력: 
"{
    "질문 요약 reasoning": "시간은 지금으로 쓰여있다. 장소는 우리집으로 쓰여있다. 전력사용량을 알고 싶은것으로 추정된다."
    "질문 요약": "지금 우리집 전력사용량 알려줘."
    "Semantic Parsing reasonsing": "지금은 2022년 9월 30일 12시 00분이다.  전력사용량은 사전정보에 없는 정보이므로 추정해야 한다."
    "Semantic Parsing": {
        "Temporal": [("지금", "2022-09-30 12:00:00")],
        "Spatial": ["우리집"],
        "Modality": ["전력사용량"],
        "Type/Quantity": ["value"],
        "Target": ["전력"],
        "Question/Actuation": ["알려줘"]
    },
    "Instruction Set": [
        ["q", "지금 우리집 전력사용량 알려줘.", "V_1"],
        ["r", "V_1", "예) '지금 우리집 전력사용량은 100kWh입니다.'"]
    ]
}"

<좋은 예시8>
입력: 어제 전원 껐어?
출력: 
"{
    "질문 요약 reasoning": "시간은 어제로 쓰여있다. 장소는 우리반으로 추측한다. 전원값을 알고 싶어하는 것으로 추정된다."
    "질문 요약": "어제 전원 값 알려줘."
    "Semantic Parsing reasonsing": "어제는 2022년 9월 29일이다."
    "Semantic Parsing": {
        "Temporal": [("어제", "2022-09-29 00:00:00 ~ 2022-09-29 23:59:59")],
        "Spatial": ["우리반"],
        "Modality": ["전원"],
        "Type/Quantity": ["value"],
        "Target": ["전원"],
        "Question/Actuation": ["알려줘"]
    },
    "Instruction Set": [
        ["q", "어제 우리반 전원 값 알려줘.", "V_1"],
        ["r", "V_1", "예) '어제 우리반 전원은 꺼져 있습니다.'"]
    ]
}"
"""

    @classmethod
    def execute(cls, input_text:str):
        """
        Change text to instruction list.
        
        Parameters:
        - input_text (str): Input text to be changed to a instruction list.
        
        Returns:
        - semantic (dict): Semantic information of the input text.
        - instructions (list[Instruction]): List of instructions.
        
        Example:
            input: 오늘 우리반과 옆반의 온도차이 알려줘.
            return:
                semantic: Semantic(
                    Temporal=[("오늘", "2022-09-30 00:00:00 ~ 2022-09-30 23:59:59")],
                    Spatial=["우리반", "옆반"],
                    Modality=["온도"],
                    Type_Quantity=["diff"],
                    Target=["온도"]
                    Question_Actuation=["알려줘"]
                ), instructions: [
                    Instruction(operation_flag="q", content="오늘 우리반과 옆반의 온도차이 알려줘", save_variable="V_1"),
                    Instruction(operation_flag="r", using_varables="V_1", example="예 ) '우리반과 옆반의 온도차이는 2도입니다'")
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
            max_new_tokens=512,
            eos_token_id=cls.terminators,
            pad_token_id=cls.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_k=5,
            top_p=0.9,
            repetition_penalty = 1.0
        )

        result_string:str = cls.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)[1:-1]
        result_dict = eval(result_string)
        
        semantic = Semantic(result_dict["Semantic Parsing"])
        instructions = [Instruction(instruction) for instruction in result_dict["Instruction Set"]]        
                
        logger.info(f"semantic: {semantic}, instructions: {instructions}")
        
        return semantic, instructions

if __name__ == "__main__":
    result = InputToInstruction.execute(
        "지난 여름 우리반과 옆반의 실내온도 비교해줘"
    )
    
    print(result)