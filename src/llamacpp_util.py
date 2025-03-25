import logging
import os
from pathlib import Path
import subprocess
import time

from src import BASE_DIR


def wait_for_input_waiting(process, logger=None):
    lines = []
    before_char = ""
    line = ""
    while process.poll() is None:
        char = process.stdout.read(1)
        if not char:
            break
        elif f"{before_char}{char}" == "\n>":
            break
        
        if char == "\n":
            if logger is not None:
                logger.info(line)
            lines.append(line)
            line = ""
        else:
            line += char

        before_char = char
    
    return "\n".join(lines).strip()

class LlamaCppModel:
    def __init__(self, 
                 model_path: Path, 
                 prompt_path: Path, 
                 gpu_config: str,
                 logger: logging.Logger,
                 grammar_path: Path = None,
                 log_output: bool = False
                 ):
        """Initialize the LlamaCppModel base class.
        
        Args:
            model_path: Path to the GGUF model file
            prompt_path: Path to the system prompt file
            binary_path: Path to the llama.cpp binary (defaults to BASE_DIR/../llama.cpp/build/bin/llama-cli)
            gpu_config: GPU tensor split configuration (default: "1,0" - primary on GPU 0)
            log_output: Whether to log model output
        """
        self.logger = logger
        self.log_output = log_output
        
        # Model paths
        self.gguf_path = model_path
        if not self.gguf_path.exists():
            self.logger.error(f"GGUF file does not exist: {self.gguf_path}")
            raise Exception("GGUF file does not exist.")
            
        self.binary_path = BASE_DIR / "../llama.cpp/build/bin/llama-cli"
        if not self.binary_path.exists():
            self.logger.error(f"Binary file does not exist: {self.binary_path}")
            raise Exception("Binary file does not exist.")
            
        self.prompt_path = prompt_path
        if not self.prompt_path.exists():
            self.logger.error(f"Prompt file does not exist: {self.prompt_path}")
            raise Exception("Prompt file does not exist.")
        
        self.grammar_path = grammar_path
        if self.grammar_path and not self.grammar_path.exists():
            self.logger.error(f"Grammar file does not exist: {self.grammar_path}")
            raise Exception("Grammar file does not exist.")
    
        # Model parameters
        self.temperature = 0.0
        self.top_p = 1.0
        self.seed = 42
        self.gpu_config = gpu_config
        
        # Process handle
        self.process = None
        
    def load(self):
        """Load the model and start the subprocess."""
        command = [
            str(self.binary_path),
            "-m", str(self.gguf_path),
            "--system-prompt-file", str(self.prompt_path),
            "-n", str(8000),  # Max tokens to generate
            "-c", str(4096),  # Context size
            "--threads", str(os.cpu_count()),
            "-ngl", str(33),
            "--temp", str(self.temperature),
            "--top_p", str(self.top_p),
            "--seed", str(self.seed),
            "--simple-io",
            "--no-display-prompt",
            "--tensor-split", self.gpu_config,
        ]
        if self.grammar_path:
            command += ["--grammar-file", str(self.grammar_path)]
        
        self.logger.info(f"Model name: {self.gguf_path.name}")
        self.logger.debug(f"Loading llama model with: {' '.join(command)}")
        
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # line-buffered
        )
        
        response = wait_for_input_waiting(self.process, self.logger if self.log_output else None)
        
        if "failed" in response:
            self.logger.error(f"Model failed to load with log: {response}")
            raise Exception("Model failed to load")
            
        return response
    
    def is_loaded(self):
        """Check if the model is loaded and the process is running."""
        is_loaded = self.process is not None and self.process.poll() is None
        if not is_loaded:
            self.ensure_loaded()
        return is_loaded
    
    def ensure_loaded(self, max_trials=3):
        """Ensure the model is loaded, attempting to reload if necessary."""
        if not self.is_loaded():
            self.logger.warning("Model is not loaded or has been closed. Attempting to load...")
            
            for trial in range(max_trials):
                try:
                    self.load()
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to load model (attempt {trial+1}/{max_trials}): {e}")
                    
            self.logger.error(f"Model failed to load after {max_trials} attempts.")
            return False
        
        return True
    
    def run_inference(self, input_text):
        """Run inference with the given input text.
        
        Args:
            input_text: The input text to process
            
        Returns:
            The model's response or None if there was an error
        """
        if not self.ensure_loaded():
            return None
            
        start_time = time.time()
        
        try:
            # Send input to the process
            self.process.stdin.write(input_text + "\n")
            self.process.stdin.flush()
            
            # Wait for and get response
            response = wait_for_input_waiting(self.process, self.logger if self.log_output else None)
            
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        end_time = time.time()
        self.logger.info(f"Inference time: {end_time - start_time:.4f}s")
        
        return response
    
    def close(self):
        """Close the model process gracefully."""
        if self.process:
            # Send interrupt signal (CTRL+C)
            self.process.send_signal(subprocess.signal.SIGINT)
            wait_for_input_waiting(self.process, self.logger if self.log_output else None)
            
            # Make sure the process terminates
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("Process didn't terminate, forcing...")
                self.process.kill()
                
            self.process = None