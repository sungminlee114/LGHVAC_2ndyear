import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
from pathlib import Path
import re
import time
from typing import List, Dict, Any

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
        max_seq_length: int = 3000,
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

    def setup_model(self, rank: int):
        """Initialize model and tokenizer for the given rank."""
        try:
            # Ensure we're using the correct device
            torch.cuda.set_device(rank)
            
            # Load model with explicit local file handling
            model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint_dir,
                torch_dtype=self.torch_dtype,
                cache_dir=self.cache_dir,
                attn_implementation=self.attn_implementation,
                load_in_4bit=self.load_in_4bit,
                local_files_only=True,
                device_map={"": rank}  # Ensure model goes to correct GPU,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint_dir,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            tokenizer.padding_side = "left"
            return model, tokenizer
            
        except Exception as e:
            print(f"Error in setup_model on rank {rank}: {str(e)}")
            raise

    @staticmethod
    def extract_content(text: str) -> str:
        """Extract content from model output."""
        pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
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
            chat = [
                {"role": "system", "content": common_prompt},
                {"role": "user", "content": f"{json.dumps(metadata)};{query}"},
            ]
            
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
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            response = tokenizer.batch_decode(outputs)[0]
            return self.extract_content(response)
            
        except Exception as e:
            print(f"Error in process_sample: {str(e)}")
            return None

    def run_rank(
        self,
        rank: int,
        world_size: int,
        dataset: List[Dict],
        common_prompt: str, 
        metadata: Dict,
        output_file: str
    ):
        """Run inference on a specific rank."""
        try:
            # Initialize the distributed process group
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            
            # Setup model and tokenizer
            model, tokenizer = self.setup_model(rank)
            
            # Calculate chunk size for this rank
            chunk_size = len(dataset) // world_size
            start_idx = rank * chunk_size
            end_idx = start_idx + chunk_size if rank != world_size - 1 else len(dataset)
            
            # Create rank-specific output file
            rank_output_file = f"{output_file}.rank{rank}"
            Path(rank_output_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Process assigned chunk
            with tqdm(
                total=end_idx - start_idx,
                desc=f"Rank {rank}",
                disable=rank != 0
            ) as pbar:
                for idx in range(start_idx, end_idx):
                    sample = dataset[idx]
                    response = self.process_sample(
                        model, tokenizer, sample["Input"],
                        common_prompt, metadata
                    )
                    
                    if response is not None:
                        
                        try:
                            response = eval(response)
                        except:
                            pass
                        
                        result = {
                            "Input": sample["Input"],
                            "Reference": sample["Response"],
                            "Candidate": response,
                        }
                        
                        with open(rank_output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    
                    pbar.update(1)
            
            # Cleanup
            dist.destroy_process_group()
            
        except Exception as e:
            print(f"Error in run_rank {rank}: {str(e)}")
            if dist.is_initialized():
                dist.destroy_process_group()
            raise

def main():
    # Configuration
    BASE_DIR = Path("../finetuning/try_lora")
    # checkpoint_dir = Path("/model/Bllossom-llama-3.2-Korean-Bllossom-3B/chkpts/r1700_a1500/checkpoint-12")
    model_name = "sh2orc-Llama-3.1-Korean-8B-Instruct"
    checkpoint_dir = Path(f"/model/{model_name}/chkpts/r256_a256/checkpoint-15")
    cache_dir = Path(f"/model/{model_name}/cache")
    
    # Verify paths exist
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
    if not BASE_DIR.exists():
        raise ValueError(f"Base directory {BASE_DIR} does not exist")
    
    # Load dataset and metadata
    dataset_path = BASE_DIR / "training_dataset_v2_directsql_ts.json"
    metadata_path = BASE_DIR / "metadata.json"
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset file {dataset_path} does not exist")
    if not metadata_path.exists():
        raise ValueError(f"Metadata file {metadata_path} does not exist")
    
    dataset = json.load(open(dataset_path, "r"))
    metadata = json.load(open(metadata_path, "r"))
    
    common_prompt = common_prompt = """
나는 훌룡한 HVAC 관련 질문 답변을 제공하는 인공지능이다. 사용자의 질문을 받아들여서, 그에 맞는 답변을 제공하는 것이 내 임무이다. 사용자의 질문을 받아들일 때, 다음과 같은 절차를 따라야 한다.
<출력 내용>
1. 'Formalized Input': 사용자의 추상적 질문을 구체화 및 정규화한 결과. 다양한 형태의 질문들을 가장 핵심적이고 근본적인 질문으로 변환한 결과.
2. 'Input Semantic Parsing': Input Semantic Parsing 결과. dict 형태로 구성되며, Temporal, Spatial, Modality, Operation을 가짐.
3. 'Strategy': 질문을 해결하기 위한 전략을 답변 전 고민. Objective: 질문의 근본적 의도 및 답변의 목적. Expected Output: 답변의 예상 결과. Step: 답변을 위한 구체으로 쪼개진 단계.
4. 'Instruction Set': 문제 해결을 위해 나의 실제 행동.

<DDL statement>
CREATE TABLE IF NOT EXISTS data_t
(
    id integer NOT NULL DEFAULT nextval('data_t_id_seq'::regclass),
    idu_id integer,
    roomtemp double precision,
    settemp double precision,
    oper boolean,
    "timestamp" timestamp without time zone NOT NULL
)
    
CREATE TABLE IF NOT EXISTS idu_t
(
    id integer NOT NULL DEFAULT nextval('idu_t_id_seq'::regclass),
    name character varying(50) COLLATE pg_catalog."default",
    metadata character varying(255) COLLATE pg_catalog."default",
    CONSTRAINT idu_t_pkey PRIMARY KEY (id)
)

"""
    
    # Get world size from available GPUs
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available")
    
    print(f"Running with {world_size} GPUs")
    
    # Initialize distributed inference
    inference = DistributedInference(
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir)
    )
    
    # Setup output file
    output_file = f"response-{model_name}.json"
    open(output_file, "w").close()  # Clear output file
    
    try:
        # Start processes
        mp.spawn(
            inference.run_rank,
            args=(world_size, dataset, common_prompt, metadata, output_file),
            nprocs=world_size,
            join=True
        )
        
        # Combine results from all ranks
        all_results = []
        for rank in range(world_size):
            rank_file = f"{output_file}.rank{rank}"
            if os.path.exists(rank_file):
                with open(rank_file, "r", encoding="utf-8") as f:
                    for line in f:
                        all_results.append(json.loads(line))
                os.remove(rank_file)
        
        # Save combined results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        # Cleanup any remaining rank files
        for rank in range(world_size):
            rank_file = f"{output_file}.rank{rank}"
            if os.path.exists(rank_file):
                os.remove(rank_file)
        raise

if __name__ == "__main__":
    main()