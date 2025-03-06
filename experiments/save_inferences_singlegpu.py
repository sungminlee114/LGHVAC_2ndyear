from unsloth import FastLanguageModel
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
from tqdm import tqdm
from datasets import Dataset
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
        load_in_4bit: bool = False,
        batch_size: int = 8  # 배치 크기 매개변수 추가
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cache_dir = Path(cache_dir)
        self.max_seq_length = max_seq_length
        self.attn_implementation = attn_implementation
        self.load_in_4bit = load_in_4bit
        self.batch_size = batch_size

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
                # from unsloth import FastLanguageModel
                
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

    def process_batch(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_data: List[Dict],
        common_prompt: str,
    ) -> str:
        try:
            batch_data = Dataset.from_list(batch_data)
            print(1, batch_data)
            
            convos = []
            for metadata, input in zip(batch_data["Metadata"], batch_data["Input"]):
                if "llama" in model.config.architectures[0].lower():
                    chat = [
                        {"role": "system", "content": common_prompt},
                        {"role": "user", "content": f"Metadata:{metadata};Input:{input};"},
                    ]
                elif "gemma" in model.config.architectures[0].lower():
                    chat = [
                        {"role": "user", "content": f"{common_prompt};{json.dumps(metadata)};{input}"},
                    ]
                else:
                    raise ValueError(f"Unsupported model architecture: {model.config.architectures[0]}")
                
                chat = tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)
                convos.append(chat)
            
            max_length = max(inputs.size(1) for inputs in convos)
        
            # 패딩 적용하여 입력 준비
            padded_inputs = []
            attention_masks = []
            
            for inputs in convos:
                pad_length = max_length - inputs.size(1)
                
                if pad_length > 0:
                    # 패딩 추가
                    padded = torch.cat([
                        torch.full((1, pad_length), tokenizer.pad_token_id, device=model.device),
                        inputs,
                    ], dim=1)
                    
                    # 어텐션 마스크 생성 (원본 시퀀스는 1, 패딩은 0)
                    mask = torch.cat([
                        torch.zeros(1, pad_length, device=model.device),
                        torch.ones(1, inputs.size(1), device=model.device),
                    ], dim=1)
                else:
                    padded = inputs
                    mask = torch.ones(1, inputs.size(1), device=model.device)
                
                padded_inputs.append(padded)
                attention_masks.append(mask)
            
            # 배치 텐서 생성
            batch_tensor = torch.cat(padded_inputs, dim=0)
            attention_mask = torch.cat(attention_masks, dim=0)
            
            # 배치 추론 실행
            outputs = model.generate(
                input_ids=batch_tensor,
                attention_mask=attention_mask,
                max_new_tokens=self.max_seq_length,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False  # 결정론적 생성
            )
            
            # 결과 디코딩 및 파싱
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            
            parsed_responses = []
            for response in responses:
                parsed = self.extract_content(response)
                if parsed is None:
                    print(f"Error parsing response: {response[:100]}...")
                    parsed_responses.append(None)
                else:
                    parsed_responses.append(parsed)
            
            return parsed_responses
            
        except Exception as e:
            print(f"Error in process_batch: {str(e)}")
            return [None] * len(batch_data)

    def run(
        self,
        dataset: List[Dict],
        common_prompt: str, 
        output_file: str
    ):
        """Run inference in batches."""
            
        # Setup model and tokenizer
        model, tokenizer = self.setup_model()
        tokenizer.padding_side = "left"
        
        # 토크나이저에 패딩 토큰 설정
        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token
        print(tokenizer.pad_token, tokenizer.eos_token)
        # 배치 처리
        with tqdm(total=len(dataset)) as pbar:
            for batch_start in range(0, len(dataset), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(dataset))
                batch_data = dataset[batch_start:batch_end]
                
                # 배치 처리
                responses = self.process_batch(
                    model, tokenizer, batch_data, common_prompt
                )
                
                # 결과 저장
                for i, response in enumerate(responses):
                    sample = batch_data[i]
                    
                    if response is not None:
                        try:
                            response = eval(response)
                        except Exception as e:
                            print(f"Error in eval: {str(e)}")
                        
                        result = {
                            "Input": sample["Input"],
                            "Scenario": sample["Scenario"],
                            "Candidate": response,
                        }
                        
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    else:
                        print(f"Error in response for sample {batch_start + i}")
                
                pbar.update(batch_end - batch_start)

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
    # checkpoint_dir = Path("/workspace/model/Bllossom-llama-3.2-Korean-Bllossom-3B/chkpts/r1700_a1500/checkpoint-12")
    

    train_type = [
        "woall", # 0
        "FI", # 1
        "ISP", # 2
        "ours" # 3
    ][1]

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
        model_name, tr_config = \
            "sh2orc-Llama-3.1-Korean-8B-Instruct", \
            f"v5_r256_a512_FI/checkpoint-43"
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

    checkpoint_dir = Path(f"/workspace/model/{model_name}/chkpts/{tr_config}")
    cache_dir = Path(f"/workspace/model/{model_name}/cache")
    
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
    batch_size = 50  # 배치 크기 설정
    inference = DistributedInference(
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir),
        batch_size=batch_size
    )
    
    # Setup output file
    output_file = f"r-{tr_config.replace('/', '-')}-batch.json"
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