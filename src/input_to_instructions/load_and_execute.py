import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Instruction:
    """
    operation_flag: choice of [
        "q" (question),
        "r" (response)
    ]
    
    -- if operation_flag is "q" --
    content: str
    save_variable: str
    variable_mapping: list[list[str, str]], [["sql_sub_question", "value"], ]
    
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
            self.value = instruction[3]
        elif self.operation_flag == "r":
            self.example:str = str(instruction[1])
    
    def __repr__(self) -> str:
        if self.operation_flag == "r":
            return f"Instruction({[self.operation_flag, self.example]})"
        else:
            return f"Instruction({[self.operation_flag, self.content, self.save_variable, self.value]})"

class Semantic:
    """
    Temporal: list of tuple
    Spatial: list of tuple
    Modality: list
    Operation: list
    Target: list
    """
    
    def __init__(self, semantic):
        self.Temporal = semantic["Temporal"]
        self.Spatial = semantic["Spatial"]
        self.Modality = semantic["Modality"]
        self.Operation = semantic["Operation"]
        self.Target = semantic["Target"]

    def __repr__(self) -> str:
        return f"Semantic(Temporal={self.Temporal}, Spatial={self.Spatial}, Modality={self.Modality}, Operation={self.Operation}, Target={self.Target})"
    
    def pformat(self) -> str:
        return f"Semantic(\n\tTemporal={self.Temporal},\n\tSpatial={self.Spatial},\n\tModality={self.Modality},\n\tOperation={self.Operation},\n\tTarget={self.Target})"
    
class InputToInstruction:
    
    PROMPT = \
"""너는 훌룡한 AI HVAC 챗봇이야. 거짓 정보들은 말하지 말아줘.
사용자의 입력에 대해 다음을 고려해 출력해줘.
출력은 무조건 서론 없이 dictionary 형태로 해줘. 다른 형태는 용납불가능해.
"이 질문에 대한 답변은 다음과 같습니다"와 같은 문장은 절대로 사용하지 말아줘.

<사전정보>
{current_metadata}

<출력 형태 및 조건>
출력은
1. 'Formalized Input': 사용자의 질문을 요약한 내용.
2. 'Input Semantic Parsing': Input Semantic Parsing 결과. dict 형태로 구성되며, Temporal, Spatial, Modality, Operation, Target key를 가짐.
3. 'Instruction Set': Instruction Set 결과. list 형태로 구성되며, 각 원소는 [flag, content, variable]로 구성됨.
출력은 무조건 예시의 dictionary 형태를 지켜줘. 무조건. 제일 중요해. 즉,시작과 끝에 무조건 중괄호가 있어야해.

<Input Semantic Parsing 조건>
Temporal: 시간 혹은 시간 범위를 나타내는 정보. 사전정보 참고. 대표적 representation과 Timestamp 형식의 튜플로 표현됨. 아무런 정보가 없을 경우 '지금'으로 표현되지만, 정보가 있는 경우 자제해야 함.
Spatial: 공간 혹은 공간 범위를 나타내는 정보. 사전정보 참고. 대표적 representation으로 표현됨. 아무런 정보가 없을 경우 '우리반'으로 표현되지만, 정보가 있는 경우 자제해야 함.
Modality: 정보의 형태 혹은 정보의 속성. 사전정보를 참고해서 제일 유사한 단어로 표현해야 하지만, 사전정보에 없는 경우 추정해야 함. 데이터의 조작, 통계, 비교등 Operation에 대한 정보는 포함되면 안됨. 예) 평균온도 -> 온도
Operation: 정보에 행해지는 조작.
Target: 정보의 대상.

<Instruction Set 조건>
공통: 'Input Semantic Parsing'의 정보를 활용하여 질문을 만들어야 함. 알 수 없음이 포함될 수 없다.
q: Query을 나타내는 flag. 두 번째 인자는 Query Generator에서 사용할 예정이며, 세 번째 인자는 query의 결과를 저장할 변수명이다.
r: Response를 나타내는 flag. 두 번째 인자는 Response를 제작하는 데 사용할 변수명이며, 세 번째 인자는 Response의 예시를 나타낸다.

<절대로 하지 말아야 할 것>
출력 앞 뒤에 불필요한 내용이 있거나 dictionary 형태를 지키지 않으면 절대로 안됨(즉, 중괄호가 무조건 있어야함). 특히 '~~질문에 대한 답변은 다음과 같습니다.나 주어진예시를~~'와 같은 문장은 절대로 작성하지 않아야 함.
이미 명시적으로 시간/공간 정보가 input에 있는데, 없다고 잘못 판단하면 안됨.
q tag 를 가진 instruction 의 경우, instruction[3] 에 sql 을 위한 세부 문장 외에 다른 문장은 넣지 말것.

<예시들과 다른 입력이 들어올 경우>
예시들과 조금 다르더라도 출력의 형식은 무조건 지켜야함.

<좋은 예시1>
입력: 오늘 우리반과 옆반의 온도차이 알려줘.
출력:
"{
    "Formalized Input": "오늘 우리반과 옆반의 실내온도 차이 알려줘.",
    "Input Semantic Parsing": {
        "Temporal": [("오늘", "2022-09-30 00:00:00 ~ 2022-09-30 23:59:59")],
        "Spatial": ["우리반", "옆반"],
        "Modality": ["실내온도"],
        "Operation": ["차이"],
        "Target": ["온도"],
    },
    "Instruction Set": [
        ["q", "오늘 우리반 실내온도 알려줘.", "우리반 실내온도", "값"],
        ["q", "오늘 옆반 실내온도 알려줘.", "옆반 실내온도", "값"],
        ["r", "예) '우리반이 옆반보다 2도 높아요'"]
    ]
}"

<좋은 예시2>
입력: 올해 여름 우리반 평균온도 알려줘.
출력: 
"{
    "Formalized Input": "올해 여름 우리반 평균 설정온도 알려줘.",
    "Input Semantic Parsing": {
        "Temporal": [("올해 여름", "2022-06-01 00:00:00 ~ 2022-08-31 23:59:59")],
        "Spatial": ["우리반"],
        "Modality": ["설정온도"],
        "Operation": ["평균"],
        "Target": ["온도"],
    },
    "Instruction Set": [
        ["q", "올해 여름 우리반 평균 설정온도 알려줘.", "올해 여름 우리반 평균 설정온도","값"]
        ["r", "예) '지난 여름 우리반의 평균 설정온도는 26도입니다.'"]
    ],
}"

<좋은 예시3>
입력: 이번달 중 우리반 온도가 가장 더운날이 언제야?
출력:
"{
    "Formalized Input": "이번달 우리반 실내온도가 최고인 날짜 알려줘.",
    "Input Semantic Parsing": {
        "Temporal": [("이번달", "2022-09-01 00:00:00 ~ 2022-09-30 23:59:59")],
        "Spatial": ["우리반"],
        "Modality": ["실내온도"],
        "Operation": ["최고"],
        "Target": ["날짜"],
    },
    "Instruction Set": [
        ["q", "이번달 우리반 실내온도가 최고인 날짜 알려줘.","이번달 우리반 실내온도가 최고인 날짜" ,"값"],
        ["r", "예) '이번달 중 우리반의 온도가 가장 더운날은 2022년 9월 15일입니다.'"]
    ]
}"

<좋은 예시4>
입력: 우리반과 옆반중에 온도가 더 낮은곳은어디야?
출력: 
"{
    "Formalized Input": "지금 우리반과 옆반 중 실내온도가 최소인 장소 알려줘.",
    "Input Semantic Parsing": {
        "Temporal": [("지금", "2022-09-30 12:00:00")],
        "Spatial": ["우리반", "옆반"],
        "Modality": ["실내온도"],
        "Operation": ["최소"],
        "Target": ["장소"],
    },
    "Instruction Set": [
        ["q", "지금 우리반 실내온도 알려줘","지금 우리반 실내온도","값"], 
        ["q","지금 옆반의 온도 알려줘 알려줘","지금 옆반의 실내온도","값"],
        ["r", "예) '우리반의 온도가 옆반보다 더 낮아요.'"]
    ]
}"

<좋은 예시5>
입력: 에어컨 켜져있어?
출력: 
"{
    "Formalized Input": "지금 우리반 전원 값 알려줘.",
    "Input Semantic Parsing": {
        "Temporal": [("지금", "2022-09-30 12:00:00")],
        "Spatial": ["우리반"],
        "Modality": ["전원"],
        "Operation": [None],
        "Target": ["전원"],
    },
    "Instruction Set": [
        ["q", "지금 우리반 전원 값 알려줘.","지금 우리반 전원 값","값"],
        ["r", "예) '지금 우리반 에어컨은 켜져 있습니다.'"]
    ]
}"

<좋은 예시6>
입력: 앞반 지금 전원 켜져있어?
출력: 
"{
    "Formalized Input": "지금 앞반 전원 값 알려줘.",
    "Input Semantic Parsing": {
        "Temporal": [("지금", "2022-09-30 12:00:00")],
        "Spatial": ["앞반"],
        "Modality": ["전원"],
        "Operation": [None],
        "Target": ["전원"],
    },
    "Instruction Set": [
        ["q", "지금 앞반 전원 값 알려줘.","지금 앞반 전원 값", "값"],
        ["r", "예) '지금 앞반 에어컨은 꺼져 있습니다.'"]
    ]
}"

<좋은 예시7>
입력: 우리집 지금 전력사용량 얼마야?
출력: 
"{
    "Formalized Input": "지금 우리집 전력사용량 알려줘.",
    "Input Semantic Parsing": {
        "Temporal": [("지금", "2022-09-30 12:00:00")],
        "Spatial": ["우리집"],
        "Modality": ["전력사용량"],
        "Operation": [None],
        "Target": ["전력"],
    },
    "Instruction Set": [
        ["q", "지금 우리집 전력사용량 알려줘.","지금 우리집 전력사용량", "값"],
        ["r", "예) '지금 우리집 전력사용량은 100kWh입니다.'"]
    ]
}"

<좋은 예시8>
입력: 어제 전원 껐어?
출력: 
"{
    "Formalized Input": "어제 전원 값 알려줘.",
    "Input Semantic Parsing": {
        "Temporal": [("어제", "2022-09-29 00:00:00 ~ 2022-09-29 23:59:59")],
        "Spatial": ["우리반"],
        "Modality": ["전원"],
        "Operation": [None],
        "Target": ["전원"],
    },
    "Instruction Set": [
        ["q", "어제 우리반 전원 값 알려줘.","어제 우리반 전원 값", "값"],
        ["r", "예) '어제 우리반 전원은 꺼져 있습니다.'"]
    ]
}"

<좋은 예시9>
입력: 지금 옆반 온도랑 우리반 온도 알려줘
출력: 
"{
    "Formalized Input": "지금 우리반과 옆반 실내온도 알려줘.",
    "Input Semantic Parsing": {
        "Temporal": [("지금", "2022-09-30 12:00:00")],
        "Spatial": ["우리반", "옆반"],
        "Modality": ["실내온도"],
        "Operation": [None],
        "Target": ["온도"],
    },
    "Instruction Set": [
        ["q","지금 우리반 실내온도 알려줘","지금 우리반 실내온도","값"], ["q","지금 옆반 실내온도 알려줘","지금 옆반 실내온도","값"],
        ["r", "예) '지금 우리반의 실내온도는 26도이고, 옆반의 실내온도는 25도입니다.'"]
    ]
}"

<좋은 예시10>
입력: 현재 옆반 온도랑 우리반 온도 알려줘
출력: 
"{
    "Formalized Input": "현재 우리반과 옆반 실내온도 알려줘.",
    "Input Semantic Parsing": {
        "Temporal": [("현재", "2022-09-30 12:00:00")],
        "Spatial": ["우리반", "옆반"],
        "Modality": ["실내온도"],
        "Operation": [None],
        "Target": ["온도"],
    },
    "Instruction Set": [
        ["q","현재 우리반 실내온도 알려줘","현재 우리반 실내온도","값"], ["q","현재 옆반 실내온도 알려줘","현재 옆반 실내온도","값"],
        ["r", "예) '현재 우리반의 실내온도는 26도이고, 옆반의 실내온도는 25도입니다.'"]
    ]
}"
"""

    @classmethod
    def execute(cls, input_text:str, current_metadata:dict):
        """
        Change text to instruction list.
        
        Parameters:
        - input_text (str): Input text to be changed to a instruction list.
        - current_metadata (dict): Metadata that may be relevant to the input text.
        
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
                    Operation=["diff"],
                    Target=["온도"]
                ), instructions: [
                    Instruction(operation_flag="q", content="오늘 우리반 온도 알려줘", save_variable="오늘 우리반 온도","값"),
                    Instruction(operation_flag="q", content="오늘 옆반 온도 알려줘", save_variable="오늘 옆반 온도","값"),
                    Instruction(operation_flag="r", example="예 ) '우리반과 옆반의 온도차이는 2도입니다'")
                ]

        """
        
        # prompt = cls.PROMPT.format(current_metadata=current_metadata)
        prompt = cls.PROMPT.replace("{current_metadata}", str(current_metadata))
        
        messages = [
            {"role": "system", "content": f"{prompt}"},
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

        result_string:str = cls.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        if result_string[0] == '"':
            result_string = result_string[1:-1]
        
        assert result_string[0] == '{'
        
        result_string = result_string.replace('\\\"', '"')
        
        try:
            result_dict = eval(result_string)
        except Exception as e:
            import traceback
            traceback.print_exception(e)
            
            print(result_string)
            
        
        semantic = Semantic(result_dict["Input Semantic Parsing"])
        instructions = [Instruction(instruction) for instruction in result_dict["Instruction Set"]]        
                
        logger.info(f"semantic: {semantic}")
        logger.info(f"instructions: {instructions}")
        
        return semantic, instructions

if __name__ == "__main__":
    from src.main import load_text_model
    
    tokenizer, model, terminators = load_text_model()
    InputToInstruction.model = model
    InputToInstruction.tokenizer = tokenizer
    InputToInstruction.terminators = terminators
    
    result = InputToInstruction.execute(
        "지난 여름 우리반과 옆반의 실내온도 비교해줘"
    )
    
    print(result)