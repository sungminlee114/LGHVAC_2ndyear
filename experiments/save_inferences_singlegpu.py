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
import subprocess
import threading

print(torch.cuda.device_count())

if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16
print(f"attn_implementation: {attn_implementation}, torch_dtype: {torch_dtype}")

import pynvml
class GPUMemoryTracker:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.peak_memory = 0
        self.track = False
    
    def tracking_loop(self):
        while self.track:
            self.update_peak_memory()
            time.sleep(.1)

    def start_tracking(self):
        # start a thread that updates the peak memory usage every 1 second
        self.track = True

        self.thread = threading.Thread(target=self.tracking_loop)
        self.thread.start()

    def stop_tracking(self):
        self.track = False
        self.thread.join()

    def get_memory_info(self):
        props = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return props.used / 1024**3 # GB
    
    def update_peak_memory(self):
        self.peak_memory = max(self.peak_memory, self.get_memory_info())
    
    def __del__(self):
        pynvml.nvmlShutdown()

class UnslothInference:
    def __init__(
        self,
        checkpoint_dir: str,
        cache_dir: str,
        max_seq_length: int = 3500,
        attn_implementation: str = attn_implementation,
        batch_size: int = 8  # 배치 크기 매개변수 추가
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cache_dir = Path(cache_dir)
        self.max_seq_length = max_seq_length
        self.attn_implementation = attn_implementation
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
                    load_in_4bit = False,
                    load_in_8bit = False,
                    # trust_remote_code=True,
                    # quantization_config=BitsAndBytesConfig(
                    #     load_in_4bit=True,
                    #     bnb_4bit_use_double_quant=True,
                    #     bnb_4bit_quant_type="nf4",
                    #     bnb_4bit_compute_dtype=torch_dtype,
                    #     # load_in_8bit=True,
                    #     # llm_int8_enable_fp32_cpu_offload=True
                    # ),
                    attn_implementation=self.attn_implementation,
                    cache_dir=self.cache_dir.as_posix(),
                    local_files_only=True,
                    device_map="cuda",
                )
                FastLanguageModel.for_inference(model)
            
            tokenizer.padding_side = "left"
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
        elif "im_start" in text:
            # <|im_start|>assistant{"Thinking": "사용자는 오늘 4층에 있는 모든 방의 설정온도의 평균값을 알고 싶어합니다. 4층에 해당하는 idu들(01_IB7, 02_I84, 02_I85)의 오늘 설정온도 데이터를 쿼리한 후 평균값을 계산하여 반환하면 됩니다.", "Expectations": ["오늘 4층의 평균 설정온도는 {{settemp_avg}}℃ 입니다."], "Instructions": [{"type": "q", "args": {"table_name": "data_t", "columns": ["settemp"], "temporal": "[DATE_TRUNC('day', DATE 'CURRENT_DATE'), DATE_TRUNC('day', DATE 'CURRENT_DATE' + INTERVAL '1 day'))", "spatials": ["01_IB7", "02_I84", "02_I85"]}, "result_name": "qr"}, {"type": "o", "script": "settemp_avg = qr['settemp'].mean();", "returns": ["settemp_avg"]}]}<|im_end|>
            pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
        elif "|endofturn|" in text:
            pattern = r"\[\|assistant\|\](.*?)\[\|endofturn\|\]"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def process_batch(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_data: List[Dict],
        common_prompt: str,
    ) -> List[Dict]:
        try:
            batch_data = Dataset.from_list(batch_data)
            
            convos = []
            start_time = time.time()
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
                    chat = [
                        {"role": "system", "content": common_prompt},
                        {"role": "user", "content": f"Metadata:{metadata};Input:{input};"},
                    ]
                    # raise ValueError(f"Unsupported model architecture: {model.config.architectures[0]}")
                
                chat = tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)
                convos.append(chat)
            
            print(convos[0])
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
            end_time = time.time()
            # print(f"Elapsed time: {end_time - start_time:.2f}s")
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
        
        print(tokenizer.pad_token, tokenizer.eos_token)
        # 배치 처리
        start_time = time.time()
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
                            "Metadata": sample["Metadata"],
                            "Candidate": response,
                        }
                        
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    else:
                        print(f"Error in response for sample {batch_start + i}")
                
                pbar.update(batch_end - batch_start)
        print(f"Elapsed time per sample: {(time.time() - start_time) / len(dataset)}s")
class LLamaInference(UnslothInference):
    def __init__(
        self,
        checkpoint_dir: str,
        gguf_path: str,
        binary_path: str = "/workspace/llama-cpp/build/bin/main",
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42
    ):
        """
        Instead of loading a model via Hugging Face, this version sets up llama.cpp inference.
        It assumes that the GGUF model file is named 'model.gguf' and is located inside checkpoint_dir.
        """
        self.gguf_path = gguf_path
        self.binary_path = binary_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir,
            local_files_only=True,
        )

        # Assume the GGUF model file is named "model.gguf" inside checkpoint_dir
        if not os.path.exists(self.gguf_path):
            raise ValueError(f"GGUF model file {self.model_path} does not exist")

    def infer(self, common_prompt:str, data: str) -> str:
        """
        Run inference by invoking the llama.cpp binary.
        Constructs a command line with the given prompt and generation parameters,
        and returns the generated text.
        """

        # if "llama" in str(self.gguf_path).lower():
        #     chat = [
        #         {"role": "system", "content": common_prompt},
        #         {"role": "user", "content": f"Metadata:{data['Metadata']};Input:{data['Input']};"},
        #     ]
        # else:
        #     raise ValueError(f"Unsupported model architecture: {self.gguf_path}")
        
        # text = str(self.tokenizer.apply_chat_template(
        #     chat,
        #     add_generation_prompt=True,
        #     tokenize=False
        # ))
        user_input = f"Metadata:{data['Metadata']};Input:{data['Input']};"
        command = [
            str(self.binary_path),
            "-m", str(self.gguf_path),
            "-sys", str(common_prompt),
            "-p", str(user_input),
            # "--chat-template", "llama3",

            "-n", str(self.max_new_tokens),
            "-c", str(len(common_prompt) + len(user_input)),
            "--threads", str(os.cpu_count()),
            "-ngl", str(33),
            "-no-cnv",
            "--temp", str(self.temperature),
            "--top_p", str(self.top_p),
            "--no-warmup",
            "--seed", str(self.seed)
        ]
        # Run the command and capture stdout
        def read_output(process):
            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # 명령어의 출력 실시간 표시
            process.stdout.close()

        print(" ".join(command))
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # 출력을 실시간으로 읽는 스레드
        thread = threading.Thread(target=read_output, args=(process,))
        thread.start()

        # 사용자의 입력을 받아서 명령어에 전달
        try:
            while process.poll() is None:  # 프로세스가 살아있는 동안
                user_input = input()  # 사용자로부터 입력 받기
                process.stdin.write(user_input + '\n')  # 프로세스에 입력 전달
                process.stdin.flush()  # 즉시 전송
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            process.stdin.close()
            process.wait()
            thread.join()

    def run(
        self,
        dataset: List[Dict],
        common_prompt: str, 
        output_file: str
    ):
        # 배치 처리
        with tqdm(total=len(dataset)) as pbar:
            for data in dataset:

                response = self.infer(common_prompt, data)
                    
                if response is not None:
                    try:
                        response = eval(response)
                    except Exception as e:
                        print(f"Error in eval: {str(e)}")
                    
                    result = {
                        "Input": data["Input"],
                        "Scenario": data["Scenario"],
                        "Metadata": data["Metadata"],
                        "Candidate": response,
                    }
                    
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    print(f"Error in response for sample {data}")
            
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
        tags = d["Tags"]["Style"]

        skip_tags = ["Reason", "Graph", "Unrelated", "Prediction"]

        skip = False
        for skip_tag in skip_tags:
            if skip_tag in tags:
                skip = True
                break
        
        if skip:
            continue

        result.append({"Metadata": metadata, "Input": d["Input"]})
    # result = [{"Input": d["Input"], "Response": json.dumps(d["Response"], ensure_ascii=False)} for d in data]
    # print(f"Read {len(result)} examples from {path}")
    # print(f"Type of result: {type(result)}")
    # print(f"Type of result[0]: {type(result[0])}")
    # print(f"Type of result[0]['Input']: {type(result[0]['Input'])}")
    # print(f"Type of result[0]['Response']: {type(result[0]['Response'])}")
    return result

def sub(name, common_prompt):
    # Remove the section between <|name|> ... <|name|> including the tags themselves
    # Use re.DOTALL to match newlines with '.'
    pattern = rf"\n?<\|{name}\|>[\s\S]*?<\|{name}\|>"
    common_prompt = re.sub(pattern, "", common_prompt, flags=re.DOTALL)
    return common_prompt

def main():
    
    # Configuration
    dataset_name = "v6-250306-optimizetoken"
    dataset_name = "v7-250309-reduceinputanddatefunctioncall"
    BASE_DIR = Path(f"../finetuning/dataset/{dataset_name}")
    

    train_type = [
        "BASE", # 0

        "5SL", # 1
        "Finetuning", # 2
        "WoThinking", # 3

        "rawSQL+LM", # 3
        "QM+script", # 4

        "NoExp", # 5
        "ours" # 6
    ][-1]

    r = 211
    model_name = "sh2orc-Llama-3.1-Korean-8B-Instruct"
    # model_name, tr_dir = \
    #     "sh2orc-Llama-3.1-Korean-8B-Instruct", \
    #     f"v7_r{r}_a{2*r}_{train_type}_tr17_0613/"

    # model_name = "LGAI-EXAONE-EXAONE-3.5-7.8B-Instruct"
    tr_dir = f"v7_r{r}_a{2*r}_{train_type}_tr27_0613"
    
    model_dir = Path(f"/model/{model_name}")
    checkpoint_dir = Path(f"{model_dir}/chkpts/{tr_dir}")

    # last checkpoint in chekpoint_dir
    checkpoint_dir = sorted(checkpoint_dir.iterdir(), key=lambda x: int(x.name.split("-")[-1]))[-1]
    tr_config = f"{tr_dir}/{checkpoint_dir.name}"
    tr_config = f"{tr_dir}/checkpoint-80"
    print(tr_config)
    checkpoint_dir = Path(f"{model_dir}/chkpts/{tr_config}")
        
    print(f"Model: {model_name}, Config: {tr_config}")

    cache_dir = Path(f"{model_dir}/cache")
    
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
    
    common_prompt = open(BASE_DIR / f"prompt.txt", "r").read()
    
    if train_type == "ours":
        sub_targets = []
    elif train_type == "BASE":
        sub_targets = ["Thinking", "Expectation", "Mapping", "Script"]
    elif train_type == "5SL":
        sub_targets = ["Thinking", "Expectation", "Mapping", "Script"]

    for sub_target in sub_targets:
        common_prompt = sub(sub_target, common_prompt)

    # remove all <||>
    common_prompt = re.sub(r"<\|.*?\|>", "", common_prompt)
    
    print(common_prompt)

    # Initialize distributed inference
    batch_size = 20  # 배치 크기 설정
    inference = UnslothInference(
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir),
        batch_size=batch_size,
        max_seq_length=10000
    )

    # gguf_path = model_dir / f"gguf/{tr_config.replace('/', '-')}.gguf"
    # if not gguf_path.exists():
    #     raise ValueError(f"GGUF model file {gguf_path} does not exist")
    # inference = LLamaInference(
    #     checkpoint_dir=checkpoint_dir,
    #     gguf_path=gguf_path,
    #     binary_path = "/workspace/LGHVAC_2ndyear/llama.cpp/build/bin/llama-cli",
    #     max_new_tokens = 500,
    # )
    
    # Setup output file
    output_file = f"r-{tr_config.replace('/', '-')}.json"
    open(output_file, "w").close()  # Clear output file
    
    track_memory = False
    if track_memory:
        memory_tracker = GPUMemoryTracker()
        memory_tracker.start_tracking()
        # add a new thread that tracks the peak memory usage

    inference.run(
        dataset=dataset,
        common_prompt=common_prompt,
        output_file=output_file
    )

    if track_memory:
        memory_tracker.stop_tracking()
        print(f"Peak memory usage: {memory_tracker.peak_memory} GB")
        print(tr_config)
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