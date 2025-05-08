import datetime
import logging

import torch

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
def extract_content(text: str) -> str:
    """Extract content from model output."""
    if "start_header_id" in text:
        pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    elif "start_of_turn" in text:
        pattern = r"<start_of_turn>model\n(.*?)<eos>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

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
            model_id = 'sh2orc/Llama-3.1-Korean-8B-Instruct'
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
                attn_implementation=attn_implementation,
                cache_dir=f"{model_dir}/cache",
                local_files_only=True,
                device_map="cuda",
            )
            FastLanguageModel.for_inference(model)
            
            tokenizer.padding_side = "left"
            cls.model, cls.tokenizer = model, tokenizer
            cls.prompt = open(prompt_path, "r").read()
    
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
    def execute(cls, instruction: InstructionR, variables, input: str, metadata: dict) -> str:
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
        if len(instruction.required_variables) == 0:
            if type(instruction.expectations) == str:
                return instruction.expectations, {}
            return instruction.expectations[0], {}
            
        # Check if all required variables are available
        if any([required_v not in variables or variables[required_v] in [] 
                for required_v in instruction.required_variables]):
            return "죄송합니다, 관련 데이터를 찾을 수 없습니다.", {}  # "Sorry, related data couldn't be found"
            
        # Filter variables to include only required ones
        result = {k: v for k, v in variables.items() if k in instruction.required_variables}
        
        # Format input for the model
        formatted_input = """질문: {input};Metadata: {metadata};예시: {expectations};결과: {result}""".format(
            input=input,
            metadata=metadata,
            expectations=instruction.expectations,
            result=result
        )
        
        # Run inference
        if cls.instance_type == "llama.cpp":
            return cls.instance.run_inference(formatted_input), result
        elif cls.instance_type == "unsloth":
            formatted_input = cls.prompt + formatted_input
            chat = cls.tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": formatted_input
                }],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(cls.model.device)
            
            outputs = cls.model.generate(
                input_ids=chat,
                max_new_tokens=1000,
                temperature=0.0,
                do_sample=False,
                pad_token_id=cls.tokenizer.eos_token_id,
            )

            responses = cls.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            response = extract_content(responses[0])

            return response, result
    
    @classmethod
    def close(cls):
        """Close the model process gracefully."""
        if cls.instance:
            cls.instance.close()