__all__ = ["InputToInstruction"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re

from src import BASE_DIR
from src.llamacpp_util import LlamaCppModel
from src.input_to_instructions.types import *

MODULE_DIR = BASE_DIR / "input_to_instructions"

class InputToInstruction:
    """Class to convert user input to instructions using a LLaMA model."""
    
    instance = None
    
    @classmethod
    def initialize(cls, train_type="ours", dtype="16bit", log_output=False):
        """Initialize the InputToInstruction class with the specified model.
        
        Args:
            train_type: The training type (default: "ours")
            dtype: The data type for the model (default: "16bit")
            log_output: Whether to log model output
        """
        if train_type != "ours":
            raise NotImplementedError("Only 'ours' is currently supported for train_type.")
            
        # Select model path based on train_type and dtype
        # gguf_path = MODULE_DIR / f"models/i2i-{train_type}-{dtype}.gguf"
        gguf_path = MODULE_DIR / f"models/v7_r256_a512_ours_16bit_adamw16bit_0324-checkpoint-56.gguf"
        prompt_path = MODULE_DIR / "prompt.txt"
        grammar_path = MODULE_DIR / "structure.gbnf"
        
        # Create LlamaCppModel instance
        cls.instance = LlamaCppModel(
            model_path=gguf_path,
            prompt_path=prompt_path,
            gpu_config="1,0",  # GPU 0 primary, GPU 1 secondary
            grammar_path=grammar_path,
            logger=logger,
            log_output=log_output
        )
        
        # Load the model
        cls.instance.load()
    
    @classmethod
    def is_loaded(cls):
        """Check if the model is loaded."""
        return cls.instance is not None and cls.instance.is_loaded()
    
    @classmethod
    def postprocess(cls, response_raw):
        # Parse the response
        try:
            if type(response_raw) != dict:
                response = eval(response_raw)
            else:
                response = response_raw
            raw_instructions = response["Instructions"]
            if "Expectations" in response:
                expectations = response["Expectations"]
            else:
                raise Exception("Expectations not found in response.")
                
        except Exception as e:
            logger.warning(f"Failed to parse response: {response_raw}. Falling to regex.")
            try:
                raw_instructions = re.search(r'(?<="Instructions": \[)(.*)(?=\])', response_raw, re.DOTALL).group(0)
                expectations = re.search(r'(?<="Expectations": \[)(.*)(?=\])', response_raw, re.DOTALL).group(0)
                if raw_instructions is None or expectations is None:
                    logger.error(f"Failed to parse response: {response_raw}.")
                    return None
            except Exception as e:
                logger.error(f"Failed to parse response: {response_raw}.")
                return None
        
        # Append response generation instruction
        types = [i["type"] for i in raw_instructions]
        if "g" not in types:
            raw_instructions.append(
                {
                    "type": "r",
                    "expectations": expectations
                }
            )
        
        # Instantiate each instruction
        instructions = []
        for raw_instruction in raw_instructions:
            try:
                match raw_instruction["type"]:
                    case "q":
                        args = raw_instruction["args"]
                        if "table_name" in args:
                            del args["table_name"]
                        # args["metadata"] = metadata

                        instructions.append(
                            InstructionQ(
                                args=args,
                                result_name=raw_instruction["result_name"]
                            )
                        )
                    case "o":
                        scripts = raw_instruction["script"].split(";")
                        scripts = [script.strip() for script in scripts]
                        scripts = [script for script in scripts if script != ""]
                        
                        instructions.append(
                            InstructionO(
                                scripts=scripts,
                                returns=raw_instruction["returns"]
                            )
                        )
                    case "g":
                        args = raw_instruction["args"]
                        plots = args["plots"]
                        required_variables = []
                        for plot in plots:
                            required_variables += [plot['data']['x']]
                            required_variables += [plot['data']['y']]
                        
                        required_variables = list(set(required_variables))

                        instructions.append(
                            InstructionG(
                                type=args["graph_type"],
                                axis=args["axis"],
                                plots=args["plots"],
                                required_variables=required_variables
                            )
                        )
                    case "r":
                        expectations = raw_instruction["expectations"]

                        # Find required variables
                        required_variables = []
                        for expectation in expectations:
                            required_variables += re.findall(r'{{(.*?)}}', expectation)
                        
                        required_variables = list(set(required_variables))
                        instructions.append(
                            InstructionR(
                                expectations=expectations,
                                required_variables=required_variables
                            )
                        )
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to parse instruction: {raw_instruction}")
                return None
        
        return instructions

    @classmethod
    def execute(cls, input_text: str, metadata: dict) -> list[InstructionQ|InstructionO|InstructionR|InstructionG|None]:
        """Convert input text to instructions.
        
        Args:
            input_text: Input text from user.
            metadata: Current metadata.
            
        Returns:
            list of instructions or None if there was an error
        """
        # Prepare the input format
        formatted_input = f"Metadata:{metadata};Input:{input_text};"
        
        # Run inference
        response_raw = cls.instance.run_inference(formatted_input)
        
        if response_raw is None:
            logger.error("Model response is None.")
            return None
            
        instructions = cls.postprocess(response_raw)
        return instructions
    
    @classmethod
    def close(cls):
        """Close the model process gracefully."""
        if cls.instance:
            cls.instance.close()