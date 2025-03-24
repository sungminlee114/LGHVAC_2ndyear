import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import subprocess
import time

from src import BASE_DIR
from src.llamacpp_util import wait_for_input_waiting
from src.input_to_instructions.types import InstructionR

MODULE_DIR = BASE_DIR / "response_generation"

class ResponseGeneration:

    @classmethod
    def load_model(
        cls,
        log_output=False,
    ):
        cls.log_output = log_output
        cls.gguf_path  = MODULE_DIR / f"models/base.gguf"
        
        if not cls.gguf_path.exists():
            logger.error(f"GGUF file does not exist: {cls.gguf_path}")
            raise Exception("GGUF file does not exist.")
        cls.binary_path = BASE_DIR / "../llama.cpp/build/bin/llama-cli"
        if not cls.binary_path.exists():
            logger.error(f"Binary file does not exist: {cls.binary_path}")
            raise Exception("Binary file does not exist.")
        cls.prompt_path = MODULE_DIR / "prompt.txt"
        if not cls.prompt_path.exists():
            logger.error(f"Prompt file does not exist: {cls.prompt_path}")
            raise Exception("Prompt file does not exist.")

        cls.temperature: float = 0.0
        cls.top_p: float = 1.0
        cls.seed: int = 42
        
        command = [
            str(cls.binary_path),
            "-m", str(cls.gguf_path),
            "--system-prompt-file", str(cls.prompt_path),
            # "-p", str(user_input),
            "-n", str(1024),
            "-c", str(1024),
            "--threads", str(os.cpu_count()),
            "-ngl", str(33),
            "--temp",   str(cls.temperature),
            "--top_p",  str(cls.top_p),
            "--seed",   str(cls.seed),
            # "-fa",
            "--simple-io",
            "--no-display-prompt",
            # "-no-cnv",
            # "-st",
            # "--no-warmup",
        ]

        logger.info(f"Model name: {cls.gguf_path.name}")
        logger.info(
            f"Loading llama model with: {' '.join(command)}"
        )

        cls.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # line-buffered
        )

        response = wait_for_input_waiting(cls.process, logger if cls.log_output else None)

        if "failed" in response:
            logger.error(f"Model failed to load with log: {response}")
            raise Exception("Model failed to load")

    @classmethod
    def run_inference(cls, input) -> str:
        trial = 0
        while cls.process.poll() is not None:
            logger.error("Model is not loaded or has been closed.")
            cls.load_model(log_output=cls.log_output)
            trial += 1
            if trial > 3:
                logger.error("Model failed to load after 3 attempts.")
                return None
        
        start_time = time.time()
        
        # send the user input to the process
        cls.process.stdin.write(input + "\n")
        cls.process.stdin.flush()

        response = wait_for_input_waiting(cls.process, logger if cls.log_output else None)
        end_time = time.time()
        logger.info(f"Time taken: {end_time - start_time}")

        return response

    @classmethod
    def execute(cls,instruction:InstructionR, variables, input: str, metadata:str) -> str:

        if len(instruction.required_variables) == 0:
            return instruction.expectations[0]
        elif any([required_v not in variables or variables[required_v] in [None] for required_v in instruction.required_variables]):
            return "죄송합니다, 관련 데이터를 찾을 수 없습니다."

        result = {k: v for k, v in variables.items() if k in instruction.required_variables}

        INPUT = """질문: {input};Metadata: {metadata};예시: {expectations};결과: {result}""".format(
            input=input,
            metadata=metadata,
            expectations=instruction.expectations,
            result=result
        )
        return cls.run_inference(INPUT)