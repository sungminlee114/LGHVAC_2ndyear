import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
from tqdm import tqdm
from pathlib import Path
import re
import time
from typing import List, Dict, Any

print(torch.cuda.device_count())

if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

class DistributedInference:
    def __init__(
        self,
        checkpoint_dir: str,
        cache_dir: str,
        max_seq_length: int = 3500,
        attn_implementation: str = attn_implementation,
        load_in_4bit: bool = False
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cache_dir = Path(cache_dir)
        self.max_seq_length = max_seq_length
        self.attn_implementation = attn_implementation
        self.load_in_4bit = load_in_4bit
        
        # Verify model files exist
        if not self.checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
        # if not (self.checkpoint_dir / "config.json").exists():
        #     raise ValueError(f"config.json not found in {checkpoint_dir}")
        
        # Set torch dtype based on GPU capability
        self.torch_dtype = torch_dtype

    def setup_model(self):
        """Initialize model and tokenizer for the given rank."""
        try:
            if False:
                # Ensure we're using the correct device
                
                # Load model with explicit local file handling
                model = AutoModelForCausalLM.from_pretrained(
                    self.checkpoint_dir,
                    torch_dtype=self.torch_dtype,
                    cache_dir=self.cache_dir.as_posix(),
                    attn_implementation=self.attn_implementation,
                    # load_in_4bit=self.load_in_4bit,
                    # load_in_8bit=True,
                    local_files_only=True,
                    device_map="cuda"
                )
                
                tokenizer = AutoTokenizer.from_pretrained(
                    self.checkpoint_dir,
                    cache_dir=self.cache_dir,
                    local_files_only=True
                )
            else:
                from unsloth import FastLanguageModel
                
                model, tokenizer = FastLanguageModel.from_pretrained(
                    self.checkpoint_dir.as_posix(),
                    dtype = self.torch_dtype,
                    # load_in_4bit = self.load_in_4bit,
                    # quantization_config=BitsAndBytesConfig(
                    #     # load_in_4bit=True,
                    #     # bnb_4bit_use_double_quant=True,
                    #     # bnb_4bit_quant_type="nf4",
                    #     # bnb_4bit_compute_dtype=torch_dtype
                    #     load_in_8bit=True,
                    #     llm_int8_enable_fp32_cpu_offload=True
                    # ),
                    attn_implementation=self.attn_implementation,
                    cache_dir=self.cache_dir.as_posix(),
                    local_files_only=True,
                    device_map="cuda",
                )
                FastLanguageModel.for_inference(model)
            return model, tokenizer
            
        except Exception as e:
            print(f"Error in setup_model {str(e)}")
            raise

    @staticmethod
    def extract_content(text: str) -> str:
        """Extract content from model output."""
        if "start_header_id" in text:
            pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
        elif "start_of_turn" in text:
            pattern = r"<start_of_turn>model\n(.*?)<eos>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def process_sample(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        query: str,
        common_prompt: str,
        metadata: Dict
    ) -> str:
        """Process a single sample using the model."""
        try:
            if "llama" in model.config.architectures[0].lower():
                chat = [
                    {"role": "system", "content": common_prompt},
                    {"role": "user", "content": f"Metadata:{metadata};Input:{query};"},
                ]
            elif "gemma" in model.config.architectures[0].lower():
                chat = [
                    {"role": "user", "content": f"{common_prompt};{json.dumps(metadata)};{query}"},
                ]
            else:
                raise ValueError(f"Unsupported model architecture: {model.config.architectures[0]}")
            
            inputs = tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=self.max_seq_length,
                use_cache=True,
            )
            
            response = tokenizer.batch_decode(outputs)[0]
            parsed_response = self.extract_content(response)
            
            if parsed_response is None:
                print(f"Error in parsing response: {response}")
            
            return parsed_response
            
        except Exception as e:
            print(f"Error in process_sample: {str(e)}")
            return None

    def run(
        self,
        dataset: List[Dict],
        common_prompt: str, 
        output_file: str
    ):
        """Run inference on a specific rank."""
            
        # Setup model and tokenizer
        model, tokenizer = self.setup_model()
        
        # Process assigned chunk
        with tqdm(
            total=len(dataset),
        ) as pbar:
            for idx in range(len(dataset)):
                sample = dataset[idx]
                response = self.process_sample(
                    model, tokenizer, sample["Input"],
                    common_prompt, sample["Metadata"]
                )
                
                if response is not None:
                    
                    try:
                        response = eval(response)
                    except Exception as e:
                        print(f"Error in eval: {str(e)}")
                        response = response
                    
                    result = {
                        "Input": sample["Input"],
                        # "Reference": sample["Response"],
                        # "Metadata": sample["Metadata"],
                        "Scenario": sample["Scenario"],
                        "Candidate": response,
                    }
                    
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    print(f"Error in response")
                pbar.update(1)

def read_dataset(train_type, dir, path):
    # the file is originally json-list format
    # we want every first-level elements to be a string itself
    # for example, [{"Hi": "a'b'"}, {"Hi": "c'd'"}] -> ["""{"Hi": "a'b'"}""", """{"Hi": "c'd'"}"""]
    
    metadata = json.load(open(dir / "metadata.json", "r"))

    path = dir / path
    with open(path, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    
    result = []
    for d in data:
        if train_type in ["woall", "FI", "ISP"]:
            del d["Response"]["Strategy"]
        
        if train_type in ["woall", "FI"]:
            del d["Response"]["Input Semantic Parsing"]
        
        if train_type in ["woall"]:
            del d["Response"]["Formalized Input"]
        
        result.append({"Metadata": metadata, "Input": d["Input"], "Response": json.dumps(d["Response"], ensure_ascii=False)})
    # result = [{"Input": d["Input"], "Response": json.dumps(d["Response"], ensure_ascii=False)} for d in data]
    # print(f"Read {len(result)} examples from {path}")
    # print(f"Type of result: {type(result)}")
    # print(f"Type of result[0]: {type(result[0])}")
    # print(f"Type of result[0]['Input']: {type(result[0]['Input'])}")
    # print(f"Type of result[0]['Response']: {type(result[0]['Response'])}")
    return result


def main():
    
    # Configuration
    BASE_DIR = Path("../finetuning/dataset/v5-250228-multimetadata")
    # checkpoint_dir = Path("/model/Bllossom-llama-3.2-Korean-Bllossom-3B/chkpts/r1700_a1500/checkpoint-12")
    

    train_type = [
        "woall", # 0
        "FI", # 1
        "ISP", # 2
        "ours" # 3
    ][0]

    if train_type == "woall":
        model_name, tr_config = \
            "sh2orc-Llama-3.1-Korean-8B-Instruct", \
            "r128_a256_woall/checkpoint-60"
        
        model_name, tr_config = \
            "sh2orc-Llama-3.1-Korean-8B-Instruct", \
            "v5_r64_a128_woall/checkpoint-72"
        # model_name, tr_config = \
        #     "sh2orc-Llama-3.1-Korean-8B-Instruct", \
        #     "r256_a512_woall/checkpoint-72"
    elif train_type == "FI":
        # model_name, tr_config = \
        #     "sh2orc-Llama-3.1-Korean-8B-Instruct", \
        #     f"r256_a512_FI/checkpoint-57"
        pass
    
    elif train_type == "ISP":
        # model_name, tr_config = \
        #     "sh2orc-Llama-3.1-Korean-8B-Instruct", \
        #     f"r256_a512_ISP/checkpoint-104"
        pass
    
    elif train_type == "ours":
        # model_name, tr_config = \
        #     "sh2orc-Llama-3.1-Korean-8B-Instruct", \
        #     "r128_a256_ours/checkpoint-51"

        # model_name, tr_config = \
        #     "sh2orc-Llama-3.1-Korean-8B-Instruct", \
        #     "r256_a512_ours/checkpoint-138"

        model_name, tr_config = \
            "sh2orc-Llama-3.1-Korean-8B-Instruct", \
            "v5_r256_a512_ours/checkpoint-54"
        
        model_name, tr_config = \
            "sh2orc-Llama-3.1-Korean-8B-Instruct", \
            "v5_r64_a128_ours/checkpoint-60"
    print(f"Model: {model_name}, Config: {tr_config}")

    checkpoint_dir = Path(f"/model/{model_name}/chkpts/{tr_config}")
    cache_dir = Path(f"/model/{model_name}/cache")
    
    # Verify paths exist
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
    if not BASE_DIR.exists():
        raise ValueError(f"Base directory {BASE_DIR} does not exist")
    
    dataset = []
    for scenario_dir in [d for d in BASE_DIR.iterdir() if d.is_dir() and "scenario" in d.name and "metadata.json" in [f.name for f in d.iterdir()]]:
        data = read_dataset(train_type, scenario_dir, "onlyq_ts.json")
        for i, d in enumerate(data):
            data[i]["Scenario"] = scenario_dir.name
        dataset.extend(data)
        
    
    common_prompt = open(BASE_DIR / F"prompt.txt", "r").read()

    if train_type in ["woall", "FI", "ISP"]:
        # search <|ST|>~~<|ST|> and remove between them
        common_prompt = re.sub(r"\n?<\|ST\|>(.|\n)*?<\|ST\|>", "", common_prompt)
    if train_type in ["woall", "FI"]:
        # search <|ISP|>~~<|ISP|> and remove between them
        common_prompt = re.sub(r"\n?<\|ISP\|>(.|\n)*?<\|ISP\|>", "", common_prompt)
    if train_type in ["woall"]:
        # search <|FI|>~~<|FI|> and remove between them
        common_prompt = re.sub(r"\n?<\|FI\|>(.|\n)*?<\|FI\|>", "", common_prompt)

    # remove all <||>
    common_prompt = re.sub(r"<\|.*?\|>", "", common_prompt)
    
    # Initialize distributed inference
    inference = DistributedInference(
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir)
    )
    
    # Setup output file
    output_file = f"response-{model_name}-{tr_config.replace('/', '-')}.json"
    open(output_file, "w").close()  # Clear output file
    
    inference.run(
        dataset=dataset,
        common_prompt=common_prompt,
        output_file=output_file
    )
    
    # Write [ at the beginning and ] at the end of the file
    # Replace \n with ,\n
    with open(output_file, "r") as f:
        lines = f.readlines()
    with open(output_file, "w") as f:
        f.write("[\n")
        f.write(",\n".join(lines))
        f.write("\n]\n")

if __name__ == "__main__":
    main()