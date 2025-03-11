import os
import json
import torch
import subprocess
from pathlib import Path
from tqdm import tqdm
import re
from typing import List, Dict, Any
from transformers import AutoTokenizer
import argparse

def extract_content(text: str) -> str:
    """Extract content from model output."""
    if "start_header_id" in text:
        pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    elif "start_of_turn" in text:
        pattern = r"<start_of_turn>model\n(.*?)<eos>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def read_dataset(train_type, dir, path):
    """Read and process dataset based on training type."""
    metadata = json.load(open(dir / "metadata.json", "r"))

    path = dir / path
    with open(path, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    
    result = []
    for d in data:
        if "v5" in dir.name:
            if train_type in ["woall", "FI", "ISP"]:
                del d["Response"]["Strategy"]
            
            if train_type in ["woall", "FI"]:
                del d["Response"]["Input Semantic Parsing"]
            
            if train_type in ["woall"]:
                del d["Response"]["Formalized Input"]
        elif "v6" in dir.name or "v7" in dir.name:
            if train_type in ["woall"]:
                del d["Response"]["생각"]

        result.append({"Metadata": metadata, "Input": d["Input"], "Response": json.dumps(d["Response"], ensure_ascii=False)})
    
    return result

def convert_lora_to_gguf(checkpoint_dir, base_model_dir, output_dir):
    """Convert LoRA checkpoint to GGUF format."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Expected output GGUF file path
    gguf_filename = f"{Path(checkpoint_dir).name}.gguf"
    output_path = Path(output_dir) / gguf_filename
    
    # Skip conversion if GGUF file already exists
    if output_path.exists():
        print(f"GGUF file {output_path} already exists, skipping conversion")
        return output_path
    
    print(f"Converting LoRA checkpoint {checkpoint_dir} to GGUF format...")
    
    # First merge the LoRA weights with the base model
    merged_dir = Path(output_dir) / "merged_model"
    os.makedirs(merged_dir, exist_ok=True)
    
    # Use transformers-cli to merge LoRA weights
    merge_cmd = [
        "python", "-m", "transformers.models.llama.convert_llama_weights_to_hf",
        "--input_dir", str(base_model_dir),
        "--model_size", "7B",  # Adjust based on your model size
        "--output_dir", str(merged_dir),
        "--load_checkpoint", str(checkpoint_dir),
    ]
    
    print(f"Running command: {' '.join(merge_cmd)}")
    subprocess.run(merge_cmd, check=True)
    
    # Convert the merged model to GGUF format using llama.cpp tools
    convert_cmd = [
        "python", "-m", "llama_cpp.convert_hf_to_gguf",
        "--outfile", str(output_path),
        "--outtype", "f16",  # Use 16-bit floats for reasonable size/performance balance
        str(merged_dir)
    ]
    
    print(f"Running command: {' '.join(convert_cmd)}")
    subprocess.run(convert_cmd, check=True)
    
    # Verify the GGUF file was created
    if not output_path.exists():
        raise RuntimeError(f"Failed to create GGUF file at {output_path}")
        
    print(f"GGUF file created at {output_path}")
    return output_path

def run_llama_cpp_inference(dataset, gguf_path, common_prompt, output_file, n_gpu_layers=35):
    """Run inference using llama.cpp with the converted GGUF model."""
    # Initialize tokenizer using the original model to help format prompts
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(gguf_path).parent / "merged_model"), 
        local_files_only=True
    )
    
    # Clear output file
    open(output_file, "w").close()
    
    # Run inference for each sample
    with tqdm(total=len(dataset)) as pbar:
        for i, sample in enumerate(dataset):
            metadata = sample["Metadata"]
            input_text = sample["Input"]
            
            # Determine if we're working with llama or gemma architecture
            # For gguf models, we'll use a simple template
            prompt = f"{common_prompt}\n\nMetadata:{json.dumps(metadata)}\nInput:{input_text}\n\nResponse:"
            
            # Create a temporary file for the prompt
            prompt_file = f"temp_prompt_{i}.txt"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)
            
            # Output file for this specific inference
            temp_output = f"temp_output_{i}.txt"
            
            # Run llama.cpp inference
            llama_cmd = [
                "llama-cpp/build/bin/main",
                "--model", str(gguf_path),
                "--file", prompt_file,
                "--n-gpu-layers", str(n_gpu_layers),
                "--ctx-size", "3500",
                "--temp", "0.0",  # Deterministic generation
                "--n-predict", "2048",  # Limit token generation
                "--output-file", temp_output,
                "--no-mmap", 
                "--mlock"
            ]
            
            print(f"Running llama.cpp inference for sample {i}...")
            subprocess.run(llama_cmd, check=True)
            
            # Read the output
            with open(temp_output, "r", encoding="utf-8") as f:
                response = f.read()
            
            # Extract the model's response
            response_text = extract_content(response)
            if response_text is None:
                # Try a simpler extraction method for llama.cpp's output
                response_text = response.split("Response:")[-1].strip()
            
            try:
                # Try to evaluate the response as JSON
                parsed_response = eval(response_text)
            except Exception as e:
                print(f"Error parsing response: {str(e)}")
                parsed_response = response_text
            
            # Store result
            result = {
                "Input": sample["Input"],
                "Scenario": sample.get("Scenario", "unknown"),
                "Metadata": sample["Metadata"],
                "Candidate": parsed_response,
            }
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            # Clean up temporary files
            os.remove(prompt_file)
            os.remove(temp_output)
            
            pbar.update(1)
    
    # Format the output file as a JSON array
    with open(output_file, "r") as f:
        lines = f.readlines()
    with open(output_file, "w") as f:
        f.write("[\n")
        f.write(",\n".join(lines))
        f.write("\n]\n")

def main():
    parser = argparse.ArgumentParser(description="Run llama.cpp inference on LoRA checkpoints")
    parser.add_argument("--dataset_name", type=str, default="v7-250309-reduceinputanddatefunctioncall", 
                        help="Dataset name")
    parser.add_argument("--model_name", type=str, default="sh2orc-Llama-3.1-Korean-8B-Instruct", 
                        help="Base model name")
    parser.add_argument("--train_config", type=str, default="v7_r256_a512_ours/checkpoint-100", 
                        help="Training configuration and checkpoint")
    parser.add_argument("--train_type", type=str, default="ours", choices=["woall", "FI", "ISP", "ours"], 
                        help="Training type")
    parser.add_argument("--n_gpu_layers", type=int, default=-1, 
                        help="Number of GPU layers to use in llama.cpp")
    args = parser.parse_args()
    
    # Set paths
    BASE_DIR = Path(f"../finetuning/dataset/{args.dataset_name}")
    checkpoint_dir = Path(f"../model/{args.model_name}/chkpts/{args.train_config}")
    base_model_dir = Path(f"../model/{args.model_name}/base")
    gguf_output_dir = Path(f"../model/{args.model_name}/gguf")
    
    # Verify paths exist
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
    if not BASE_DIR.exists():
        raise ValueError(f"Base directory {BASE_DIR} does not exist")
    if not base_model_dir.exists():
        raise ValueError(f"Base model directory {base_model_dir} does not exist")
    
    # Load dataset
    dataset = []
    for scenario_dir in [d for d in BASE_DIR.iterdir() if d.is_dir() and "scenario" in d.name and "metadata.json" in [f.name for f in d.iterdir()]]:
        data = read_dataset(args.train_type, scenario_dir, "onlyq_ts.json")
        for i, d in enumerate(data):
            data[i]["Scenario"] = scenario_dir.name
        dataset.extend(data)
    
    # Load and process prompt
    common_prompt = open(BASE_DIR / "prompt.txt", "r").read()
    
    # Process prompt based on training type
    if "v5" in BASE_DIR.name:
        if args.train_type in ["woall", "FI", "ISP"]:
            common_prompt = re.sub(r"\n?<\|ST\|>(.|\n)*?<\|ST\|>", "", common_prompt)
        if args.train_type in ["woall", "FI"]:
            common_prompt = re.sub(r"\n?<\|ISP\|>(.|\n)*?<\|ISP\|>", "", common_prompt)
        if args.train_type in ["woall"]:
            common_prompt = re.sub(r"\n?<\|FI\|>(.|\n)*?<\|FI\|>", "", common_prompt)
    elif "v6" in BASE_DIR.name or "v7" in BASE_DIR.name:
        if args.train_type in ["woall"]:
            common_prompt = re.sub(r"\n?<\|Ours\|>(.|\n)*?<\|Ours\|>", "", common_prompt)
    
    # Remove all <||> tags
    common_prompt = re.sub(r"<\|.*?\|>", "", common_prompt)
    
    # Convert LoRA checkpoint to GGUF format
    gguf_path = convert_lora_to_gguf(checkpoint_dir, base_model_dir, gguf_output_dir)
    
    # Setup output file
    output_file = f"llama-cpp-{args.train_config.replace('/', '-')}.json"
    
    # Run inference using llama.cpp
    run_llama_cpp_inference(
        dataset=dataset,
        gguf_path=gguf_path,
        common_prompt=common_prompt,
        output_file=output_file,
        n_gpu_layers=args.n_gpu_layers
    )
    
    print(f"Inference completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()