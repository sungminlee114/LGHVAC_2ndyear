import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src import BASE_DIR
from src.llamacpp_util import LlamaCppModel
from src.input_to_instructions.types import InstructionR

MODULE_DIR = BASE_DIR / "response_generation"

class ResponseGeneration:
    """Class to generate responses using a LLaMA model."""
    
    instance = None
    
    @classmethod
    def initialize(cls, log_output=False):
        """Initialize the ResponseGeneration class.
        
        Args:
            log_output: Whether to log model output
        """
        # Model paths
        gguf_path = MODULE_DIR / "models/base.gguf"
        prompt_path = MODULE_DIR / "prompt.txt"
        
        # Create LlamaCppModel instance
        cls.instance = LlamaCppModel(
            model_path=gguf_path,
            prompt_path=prompt_path,
            gpu_config="0,1",  # GPU 1 primary, GPU 0 secondary (opposite of InputToInstruction)
            logger=logger,
            log_output=log_output
        )
        
        cls.instance.load()
    
    @classmethod
    def is_loaded(cls):
        """Check if the model is loaded."""
        return cls.instance is not None and cls.instance.is_loaded()
    
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
        return cls.instance.run_inference(formatted_input), result
    
    @classmethod
    def close(cls):
        """Close the model process gracefully."""
        if cls.instance:
            cls.instance.close()