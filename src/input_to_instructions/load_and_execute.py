__all__ = ["InputToInstruction"]

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import subprocess
import os
import time
import re

from src import BASE_DIR
from src.llamacpp_util import wait_for_input_waiting
from src.input_to_instructions.types import *

MODULE_DIR = BASE_DIR / "input_to_instructions"

class InputToInstruction:
    @classmethod
    def parse_arg(self,
        train_type:str,
        dtype:str,
        log_output=False,
    ):
        self.train_type = train_type
        if self.train_type != "ours":
            raise NotImplementedError("Only 'ours' is currently supported for train_type.")

        self.dtype = dtype
        self.log_output = log_output

        self.gguf_path  = MODULE_DIR / f"models/i2i-{self.train_type}-{self.dtype}.gguf"
        if not self.gguf_path.exists():
            logger.error(f"GGUF file does not exist: {self.gguf_path}")
            raise Exception("GGUF file does not exist.")
        self.binary_path = BASE_DIR / "../llama.cpp/build/bin/llama-cli"
        if not self.binary_path.exists():
            logger.error(f"Binary file does not exist: {self.binary_path}")
            raise Exception("Binary file does not exist.")
        self.prompt_path = MODULE_DIR / "prompt.txt"
        if not self.prompt_path.exists():
            logger.error(f"Prompt file does not exist: {self.prompt_path}")
            raise Exception("Prompt file does not exist.")

        self.temperature: float = 0.0
        self.top_p: float = 1.0
        self.seed: int = 42

    @classmethod
    def load_model(
            self,
            train_type,
            dtype,
            log_output=False
        ):

        self.parse_arg(
            train_type=train_type,
            dtype=dtype,
            log_output=log_output
        )

        command = [
            str(self.binary_path),
            "-m", str(self.gguf_path),
            "--system-prompt-file", str(self.prompt_path),
            # "-p", str(user_input),
            "-n", str(4096),
            "-c", str(1024),
            "--threads", str(os.cpu_count()),
            "-ngl", str(33),
            "--temp",   str(self.temperature),
            "--top_p",  str(self.top_p),
            "--seed",   str(self.seed),
            # "-fa",
            "--simple-io",
            "--no-display-prompt",
            # "--sm", "none",
            # "--main-gpu", "0",
            "--tensor-split", "1,0", # gpu 0, 1
            # "-no-cnv",
            # "-st",
            # "--no-warmup",
        ]
        # open a new process to run the command

        logger.info(f"Model name: {self.gguf_path.name}")
        logger.debug(
            f"Loading llama model with: {' '.join(command)}"
        )

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # line-buffered
        )

        response = wait_for_input_waiting(self.process, logger if self.log_output else None)

        if "failed" in response:
            logger.error(f"Model failed to load with log: {response}")
            raise Exception("Model failed to load")
    
    @classmethod
    def run_inference(self, input_text:str, metadata:dict):
        
        trial = 0
        while self.process.poll() is not None:
            logger.error("Model is not loaded or has been closed.")
            self.load_model(
                train_type=self.train_type,
                dtype=self.dtype,
                log_output=self.log_output
            )
            trial += 1
            if trial > 3:
                logger.error("Model failed to load.")
                return None
        
        start_time = time.time()
        
        try:
            user_input = f"Metadata:{metadata};Input:{input_text};"

            # send the user input to the process
            self.process.stdin.write(user_input + "\n")
            self.process.stdin.flush()

            response = wait_for_input_waiting(self.process, logger if self.log_output else None)
        except Exception as e:
            logger.error(f"Error while i2i: {user_input}, {e}")
            import traceback
            traceback.print_exc()
            return None

        end_time = time.time()
        logger.info(f"Time taken: {end_time - start_time}")

        return response

    @classmethod
    def execute(self, input_text:str, metadata:dict) -> list[InstructionQ|InstructionO|InstructionR|None]:
        """ Convert input text to instructions.

        Args:
            input_text (str): Input text from user.
            metadata (dict): Current metadata.

        Returns:
            instructions (list[InstructionQ|InstructionO|InstructionR]): List of instructions.
        """

        response_raw = self.run_inference(input_text, metadata)

        if response_raw is None:
            logger.error("Model response is None.")
            return None

        # Check if response is a valid json
        try:
            response = eval(response_raw)
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
            except Exception as e:
                logger.error(f"Failed to parse response: {response_raw}.")
                return None
        
        # Append response generation instruction
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
                        del args["table_name"]
                        args["metadata"] = metadata

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
                        instructions.append(
                            InstructionG(
                                type=raw_instruction["graph_type"],
                                axis=raw_instruction["axis"],
                                plots=raw_instruction["plots"]
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
    def observe_closing(self):
        # change stdout and stderr to default values
        # self.process.stdout = None
        # self.process.stderr = NoneInputToInstruction

        # send ^C to the process
        self.process.send_signal(subprocess.signal.SIGINT)
        wait_for_input_waiting(self.process, logger)
        
        # self.process.terminate()
        # self.process.wait()

if __name__ == "__main__":

    InputToInstruction.load_model(
        train_type=["woall", "ours"][1],
        dtype=["F16", "Q8_0"][1],
        log_output=False
    )

    current_metadata = {
        "site_name": "YongDongIllHighSchool",
        "user_name": "홍길동", "user_role": "customer", "idu_name": "01_IB5",
        "idu_mapping": {"01_IB5": ["우리반"], "01_IB7": ["옆반"], "02_I81": ["앞반"]},
        "modality_mapping": {"roomtemp": ["실내온도"], "settemp": ["설정온도"], "oper": ["전원"]},
        "current_datetime": "2022-09-30 12:00:00"
    }

    for _ in range(1):
        input = "지난 여름 우리반과 옆반의 실내온도 비교해줘"
        response = InputToInstruction.execute(input, current_metadata)
        print(response)

    InputToInstruction.observe_closing()