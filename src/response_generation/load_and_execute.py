import datetime
import logging

import torch
from unsloth.models.llama import BitsAndBytesConfig

from src.db import instance
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from unsloth import FastLanguageModel
import pandas as pd

from src import BASE_DIR
from src.llamacpp_util import LlamaCppModel
from src.input_to_instructions.types import InstructionR

MODULE_DIR = BASE_DIR / "response_generation"

import re
def extract_content(text: str) -> str | None:
    """Extract content from model output."""
    pattern: str | None = None
    if "start_header_id" in text:
        pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    elif "start_of_turn" in text:
        pattern = r"<start_of_turn>model\n(.*?)<eos>"
    elif "|endofturn|" in text:
        pattern = r"\[\|assistant\|\](.*?)\[\|endofturn\|\]"
    if pattern:
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    return None

class ResponseGeneration:
    """Class to generate responses using a LLaMA model."""
    
    instance = None

    
    @classmethod
    def initialize(cls, log_output=False, instance_type="llama.cpp"):
        """Initialize the ResponseGeneration class.
        
        Args:
            log_output: Whether to log model output
        """
        # Model paths
        gguf_path = MODULE_DIR / "models/base.gguf"
        prompt_path = MODULE_DIR / "prompt.txt"
        
        # Create LlamaCppModel instance
        cls.instance_type = instance_type
        if instance_type == "llama.cpp":
            cls.instance = LlamaCppModel(
                model_path=gguf_path,
                prompt_path=prompt_path,
                gpu_config="0,1",  # GPU 1 primary, GPU 0 secondary (opposite of InputToInstruction)
                logger=logger,
                log_output=log_output
            )
            cls.instance.load()
        elif instance_type == "unsloth":
            # model_id = 'sh2orc/Llama-3.1-Korean-8B-Instruct'
            # model_id = 'Bllossom/llama-3-Korean-Bllossom-70B'
            model_id = 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct'
            model_dir = f"/model/{model_id.replace('/', '-')}"
            if torch.cuda.get_device_capability()[0] >= 8:
                attn_implementation = "flash_attention_2"
                torch_dtype = torch.bfloat16
            else:
                attn_implementation = "eager"
                torch_dtype = torch.float16
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_id,
                dtype = torch_dtype,
                load_in_4bit = False,
                load_in_8bit = False,
                trust_remote_code=True,
                # quantization_config=BitsAndBytesConfig(
                #         # load_in_4bit=True,
                #         # bnb_4bit_use_double_quant=True,
                #         # bnb_4bit_quant_type="nf4",
                #         # bnb_4bit_compute_dtype=torch_dtype,
                #         load_in_8bit=True,
                #         # llm_int8_enable_fp32_cpu_offload=True
                #     ),
                attn_implementation=attn_implementation,
                cache_dir=f"{model_dir}/cache",
                # local_files_only=True,
                device_map="cuda",
            )
            FastLanguageModel.for_inference(model)
            
            tokenizer.padding_side = "left"
            cls.model, cls.tokenizer = model, tokenizer
            cls.prompt = open(prompt_path, "r").read()
    
    @classmethod
    def update_prompt(cls):
        cls.prompt = open(MODULE_DIR / "prompt.txt", "r").read()

    @classmethod
    def is_loaded(cls):
        """Check if the model is loaded."""
        return cls.instance is not None and cls.instance.is_loaded()
    
    @classmethod
    def stringify_variables(cls, variables: dict) -> dict:
        # there are some variables that are pd.DataFrame, pd.Series, timestamp
        # we need to convert them to string
        for k, v in variables.items():
            # print(type(v))
            if isinstance(v, pd.DataFrame):
                # print([v[col].dtype for col in v.columns])
                for col in v.columns:
                    if v[col].dtype == "datetime64[ns]":
                        v[col] = v[col].astype(str)
                    elif col == "date":
                        try:
                            # Series of datetime.date
                            v[col] = v[col].apply(lambda x: x.strftime("%Y-%m-%d"))
                        except:
                            # Might already be a string
                            pass
                variables[k] = v.to_dict(orient="records")
            elif isinstance(v, pd.Series):
                variables[k] = v.to_dict()
            elif isinstance(v, pd.Timestamp):
                variables[k] = v.strftime("%Y-%m-%d %H:%M:%S")
            else:
                variables[k] = str(v)
        return variables
    
    @classmethod
    def execute_raw(cls, input: str, prompt=None) -> str:
        "unsloth"

        chat = []
        if prompt is not None:
            chat.append({
                "role": "system",
                "content": prompt
            })
        chat.append({
            "role": "user",
            "content": input
        })
        chat = cls.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(cls.model.device)

        outputs = cls.model.generate(
            input_ids=chat,
        )

        response = cls.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        return response[0]
    
    @classmethod
    def execute_v2(cls, expectations: list[str], required_variables: list[str], variables: dict, input: str, exp_tag=None) -> tuple[str | None, dict]:
        if exp_tag not in ["woExp"] and len(required_variables) == 0:
            return expectations[0], {}
        
        if exp_tag in ["woExp", "woScript", "woQM+Script"]:
            result = variables
        else:
            result = {k: v for k, v in variables.items() if k in required_variables}
        formatted_input = """질문: {input}; 포맷: {expectations}; 데이터: {result};""".format(
            input=input,
            expectations=expectations,
            result=result
        )
        # print(formatted_input)
        chat = cls.tokenizer.apply_chat_template(
            [{
                "role": "system",
                "content": cls.prompt
            },
            {
                "role": "user",
                "content": formatted_input
            }],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(cls.model.device)

        # measure_token_count = lambda input: len(tokenizer.encode(str(input)))
        cls.last_input_str= formatted_input
        # last_input_token_length = measure_token_count(
        #     cls.tokenizer.apply_chat_template(
        #         [{
        #             "role": "system",
        #             "content": cls.prompt
        #         },
        #         {
        #             "role": "user",
        #             "content": formatted_input
        #         }]
        #     )
        # )
        # 모델 추론을 실행합니다.
        outputs = cls.model.generate(
            input_ids=chat,
            max_new_tokens=1000,
            # temperature=0.0,
            do_sample=False,
            pad_token_id=cls.tokenizer.eos_token_id,
        )

        responses = cls.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        response = extract_content(responses[0])

        cls.last_output_str = response
        return response, result
    
    @classmethod
    def execute(cls, instruction: InstructionR, variables, input: str, metadata: dict | None, exp_tag=None) -> tuple[str | None, dict]:
        """Generate a response based on the given instruction, variables and input.
        
        Args:
            instruction: The response instruction
            variables: Dict containing variable values
            input: The original user input
            metadata: The metadata dict
            
        Returns:
            The generated response
        """
        # If no variables are required, return the first expectation directly
        if exp_tag is None and len(instruction.required_variables) == 0:
            if isinstance(instruction.expectations, str):
                return instruction.expectations, {}
            return instruction.expectations[0], {}
            
        # # Check if all required variables are available
        # if exp_tag == None and any([required_v not in variables or variables[required_v] in [] 
        #         for required_v in instruction.required_variables]):
        #     return "죄송합니다, 관련 데이터를 찾을 수 없습니다.", {}  # "Sorry, related data couldn't be found"
            
        # Filter variables to include only required ones
        if exp_tag in ["woCoTExp", "woOp"]:
            result = {k: v for k, v in variables.items() if k not in ["Metadata"]}
        else:
            result = {k: v for k, v in variables.items() if k in instruction.required_variables}

        # Format input for the model
        if metadata is None:
            formatted_input = """질문: {input}; 포맷: {expectations}; 데이터: {result};""".format(
                input=input,
                expectations=instruction.expectations,
                result=result
            )
        else:
            if exp_tag != "woCoTExp":
                formatted_input = """질문: {input}; Metadata: {metadata}; 예시: {expectations}; 데이터: {result}""".format(
                    input=input,
                    metadata=metadata,
                    expectations=instruction.expectations,
                    result=result
                )
            else:
                formatted_input = """질문: {input}; Metadata: {metadata}; 데이터: {result}""".format(
                    input=input,
                    metadata=metadata,
                    result=result
                )
        
        # Run inference
        if cls.instance_type == "llama.cpp":
            assert cls.instance is not None
            return cls.instance.run_inference(formatted_input), result
        elif cls.instance_type == "unsloth":
            # print(instruction.expectations, result)
            print(formatted_input)
            # formatted_input = cls.prompt + formatted_input
            chat = cls.tokenizer.apply_chat_template(
                [{
                    "role": "system",
                    "content": cls.prompt
                },
                {
                    "role": "user",
                    "content": formatted_input
                }],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(cls.model.device)

            # model의 입력 dtype을 가져와서 chat 텐서에 맞춰줍니다.
            # 일반적으로 model의 첫 번째 파라미터의 dtype을 사용합니다.
            # chat = chat.to(device=cls.model.device)

            # 모델 추론을 실행합니다.
            outputs = cls.model.generate(
                input_ids=chat,
                max_new_tokens=1000,
                # temperature=0.0,
                do_sample=False,
                pad_token_id=cls.tokenizer.eos_token_id,
            )

            responses = cls.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            response = extract_content(responses[0])

            return response, result
        return None, {}
    
    @classmethod
    def close(cls):
        """Close the model process gracefully."""
        if cls.instance:
            cls.instance.close()