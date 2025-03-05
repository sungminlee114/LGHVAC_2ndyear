# %%
import torch
print(torch.cuda.device_count())

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from unsloth import FastLanguageModel
import pprint
import json
from pathlib import Path
import transformers
import os


# %%
train_type = [
    "woall", # 0
    "FI", # 1
    "ISP", # 2
    "ours" # 3
][3]
print(f"train_type: {train_type}")

# %%
BASE_DATASET_DIR = Path("../dataset/v5-250228-multimetadata")
print(f"BASE_DATASET_DIR: {BASE_DATASET_DIR}")
print(list(BASE_DATASET_DIR.iterdir()))

# %%
max_seq_length = 0     # Unsloth auto supports RoPE Scaling internally!
dtype = None              # None for auto detection
load_in_4bit = False      # Use 4bit quantization to reduce memory usage. Can be False.
device = f"cuda"



if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16
# attn_implementation = "eager"
print(f"Using {attn_implementation} for attention computation.")
# QLora?

# %%
import re

# current_metadata = json.load(open("metadata.json", "r"))


common_prompt = open(BASE_DATASET_DIR / f"prompt.txt", "r").read()

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
print(common_prompt)

# print(common_prompt)

# %%
model_id = 'sh2orc/Llama-3.1-Korean-8B-Instruct'
# model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
# model_id = 'Saxo/Linkbricks-Horizon-AI-Korean-Gemma-2-sft-dpo-27B'

model_dir = f"/model/{model_id.replace('/', '-')}"

# %% [markdown]
# ## Load tokenizer and dataset

# %%
# Tokenizer initialization
pretrained_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,  
    # max_seq_length = max_seq_length,
    dtype = dtype,
    # load_in_4bit = False if not "27B" in model_id else True,
    # quantization_config=BitsAndBytesConfig(
    #     # load_in_4bit=True,
    #     # bnb_4bit_use_double_quant=True,
    #     # bnb_4bit_quant_type="nf4",
    #     # bnb_4bit_compute_dtype=torch_dtype
    #     load_in_8bit=False if not "27B" in model_id else True,
    #     llm_int8_enable_fp32_cpu_offload=False if not "27B" in model_id else True,
    # ),
    # device_map=device,
    cache_dir=f"{model_dir}/cache",
    attn_implementation=attn_implementation,
    # local_files_only=True
)

# if not os.path.exists(model_dir):
# pretrained_model.save_pretrained(model_dir)

# %%
tokenizer.padding_side = "left"
# tokenizer.truncation_side = "left"
print(f"Pad Token id: {tokenizer.pad_token_id} and Pad Token: {tokenizer.pad_token}")
print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")

# %%
scenario_dirs = [d for d in BASE_DATASET_DIR.iterdir() if d.is_dir() and "scenario" in d.name and "metadata.json" in [f.name for f in d.iterdir()]]
print(scenario_dirs)

# %%

def read_dataset(dir, path):
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

dataset_trs = []
dataset_tss = []
for scenario_dir in scenario_dirs:
    dataset_trs.extend(read_dataset(scenario_dir, "onlyq_tr.json"))
    dataset_tss.extend(read_dataset(scenario_dir, "onlyq_ts.json"))

dataset_tr = Dataset.from_list(dataset_trs)
dataset_ts = Dataset.from_list(dataset_tss)

max_seq_length = 0
def formatting_prompts_func(examples):
    convos = []
    # Iterate through each item in the batch (examples are structured as lists of values)
    for metadata, input, response in zip(examples['Metadata'], examples['Input'], examples['Response']):
        global max_seq_length
        response.replace("    ", "")

        answer = {
            "content": f"{response}",
            "role": "assistant"
        }
        if "llama" in model_id.lower():
            prompt = {
                "content": common_prompt,
                "role": "system"
            }
            user_input = {
                "content": f"Metadata:{metadata};Input:{input};",
                "role": "user"
            }
            convos.append([prompt, user_input, answer])
        elif "gemma" in model_id.lower():
            user_input = {
                "content": f"{common_prompt};{metadata};{input}",
                "role": "user"
            }
            convos.append([user_input, answer])
        
        
        
        
        # print("Answer length: ", len(response))
        # convos.append([prompt, user_input, answer])
        
        if len(response) + 50 > max_seq_length:
            max_seq_length = len(response) + len(metadata) + len(input) + 50
            # print(response)
    
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos]
    # remove \n\nCutting Knowledge Date: BLAH BLAH \nToday Date: BLAH BLAH\n\n using regex
    texts = [re.sub(r'(\nCutting Knowledge Date:.*?\nToday Date:.*?\n\n)', '', text) for text in texts]
    
    return {"text": texts}

dataset_tr = dataset_tr.map(formatting_prompts_func, batched=True)
dataset_ts = dataset_ts.map(formatting_prompts_func, batched=True)

print(dataset_tr[0]["text"])
max_seq_length += len(common_prompt)
print(max_seq_length)
# print(f"seq length: {len(tokenizer.encode(dataset_tr[0]['text']))}")

# %%
lora_r = 64
lora_alpha = 128
lora_repr = f"v5_r{lora_r}_a{lora_alpha}_{train_type}"
print(lora_repr)

# %%


peft_model = FastLanguageModel.get_peft_model(
    pretrained_model,
    r=lora_r,   # LoRA rank - suggested values: 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj",
                    # "embed_tokens", 
                    # "lm_head"
                    ],
    lora_alpha=lora_alpha,
    lora_dropout=0.05,   # Supports any, but = 0 is optimized
    bias="none",      # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # Ideal for long context tuning
    random_state=3407,
    use_rslora=False,   # Disable rank-sensitive LoRA for simpler tasks
    loftq_config=None   # No LoftQ, for standard fine-tuning
)
# del pretrained_model


# %% [markdown]
# ## Training config

# %% [markdown]
# ## Train

# %%

torch.cuda.empty_cache()
print(len(dataset_tr))


# %%
import numpy as np


per_device_train_batch_size, epochs = 37, 70 # 8
gradient_accumulation_steps = int(np.ceil(len(dataset_tr) / per_device_train_batch_size))
print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")

# clear all checkpoints
import shutil
shutil.rmtree(f"{model_dir}/chkpts/{lora_repr}", ignore_errors=True)

args = TrainingArguments(
    # num_train_epochs = 1,
    per_device_train_batch_size = per_device_train_batch_size,  # Controls the batch size per device
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,  # Accumulates gradients to simulate a larger batch
    max_steps=gradient_accumulation_steps * epochs,
    # 리소스 제약때문에 batch size를 타협해야하는 경우가 발생 -> micro batch size를 줄이고,
 	# accumulated step을 늘려, 적절한 size로 gradient를 구해 weight update
    # https://www.youtube.com/watch?v=ptlmj9Y9iwE
    warmup_steps = gradient_accumulation_steps,
    learning_rate = 1e-4,             # Sets the learning rate for optimization
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    optim = "adamw_8bit",
    weight_decay = 0.01,              # Regularization term for preventing overfitting
    lr_scheduler_type = "cosine",  # Sets the learning rate scheduler
    seed = 3407,                        
    output_dir = f"{model_dir}/chkpts/{lora_repr}",  # Output directory for checkpoints and predictions     
    report_to = "none",              # Enables Weights & Biases (W&B) logging
    logging_steps = gradient_accumulation_steps,                # Sets frequency of logging to W&B
    logging_strategy = "steps",       # Logs metrics at each specified step
    evaluation_strategy="steps",  # enable evaluation during training
    eval_steps=gradient_accumulation_steps,
    # eval_accumulation_steps=1, # 낮을수록 eval시 사용하는 메모리 줄어듦
    save_steps=gradient_accumulation_steps,
    save_strategy = "steps",               
    load_best_model_at_end = True,    # Loads the best model at the end
    save_only_model = False           # Saves entire model, not only weights
)

from trl import SFTTrainer

trainer = SFTTrainer(
    model = peft_model,
    processing_class = tokenizer,
    train_dataset = dataset_tr,
    eval_dataset = dataset_ts,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,        # Can make training 5x faster for short sequences.
    args = args,
)

# %%
from unsloth import unsloth_train

trainer_stats = unsloth_train(trainer)
print(trainer_stats)


# %%
# del base_model
# del peft_model

torch.cuda.empty_cache()

# Specify the checkpoint directory
checkpoint_dir = f"{model_dir}/chkpts/{lora_repr}/checkpoint-{18}"
# checkpoint_dir = "/model/Bllossom-llama-3.2-Korean-Bllossom-3B/chkpts/r1700_a1500/checkpoint-12"
print(checkpoint_dir)
# Load the tokenizer (ensure it's the same tokenizer used for training)

# Load the base model
peft_model, tokenizer = FastLanguageModel.from_pretrained(
    checkpoint_dir,
    dtype = dtype,
    # max_seq_length = max_seq_length,
    # quantization_config=BitsAndBytesConfig(
    #     # load_in_4bit=True,
    #     # bnb_4bit_use_double_quant=True,
    #     # bnb_4bit_quant_type="nf4",
    #     # bnb_4bit_compute_dtype=torch_dtype
    #     load_in_8bit=True,
    #     llm_int8_enable_fp32_cpu_offload=True
    # ),
    device_map=device,
    attn_implementation=attn_implementation,
    cache_dir=f"{model_dir}/cache",
    local_files_only=True
)

# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
# tokenizer.truncation_side = "left"

print(f"Pad Token id: {tokenizer.pad_token_id} and Pad Token: {tokenizer.pad_token}")
print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")
print(f"Padding side: {tokenizer.padding_side}")
# tokenizer.pad_token = tokenizer.eos_token
# 


# Make sure the tokenizer is ready


# %%
if False:
    # Local saving
    peft_model.save_pretrained("lora_i2i") 
    tokenizer.save_pretrained("lora_i2i")

    # For merging the LoRA adapters with the base model and save the model to 16-bit precision for optimized performance with vLLM, use:
    # # Merge to 16bit
    peft_model.save_pretrained_merged("i2i_merged_16bit", tokenizer, save_method = "merged_16bit",)
    peft_model.save_pretrained_merged("i2i_merged_4bit", tokenizer, save_method = "merged_4bit_forced",)
    # model.push_to_hub_merged("<hf_username/model_name>", tokenizer, save_method = "merged_16bit", token = hf_token)

# %% [markdown]
# ## Tryout

# %%
from unsloth import FastLanguageModel
from transformers import TextStreamer
if False:
    # del peft_model
    torch.cuda.empty_cache()
    peft_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "i2i_merged_16bit",        # Trained model either locally or from huggingface
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = False,
        attn_implementation=attn_implementation,
        # device_map=[device],
        local_files_only=True
    )
FastLanguageModel.for_inference(peft_model)  # Enable native 2x faster inference


# %% [markdown]
# [Instruction(['q', '이번달 우리반 실내온도가 최고인 날짜 알려줘.', '이번달 우리반 실내온도가 최고인 날짜', '값', Semantic(Temporal=[('이번달', '2022-09-01 00:00:00 ~ 2022-09-30 23:59:59')], Spatial=['우리반'], Modality=['실내온도'], Operation=['최고인'], Target=['날짜'])]), Instruction(['r', "예) '이번달 우리반의 실내 온도가 가장 더웠던 날은 2022년 9월 15일입니다.'"])]

# %%
import re, time

def extract_content(text):
    # Define the regex pattern to extract the content
    print(text)
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def run(query):
    start_time = time.time()
    chat = [
        {"role": "system", "content": common_prompt},
        {"role": "user", "content": f"Metadata:{current_metadata};Input:{query};"},
    ]
    
    inputs = tokenizer.apply_chat_template(
        chat,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")
    
    outputs = peft_model.generate(
        input_ids = inputs,
        max_new_tokens = max_seq_length,
        use_cache = True,
        pad_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id,
    )
    
    response = tokenizer.batch_decode(outputs)[0]
    # print(response)
    # print("Extracting..")
    parsed_response = extract_content(response)
    pprint.pprint(f"Query: {query}, Time: {time.time() - start_time}")
    # print(parsed_response)
    
    parsed_response_dict = eval(parsed_response)
    
    return parsed_response, parsed_response_dict
    
    # text_streamer = transformers.TextStreamer(
    #     tokenizer, 
    # skip_prompt = True
    # )
    # _ = peft_model.generate(
    #     input_ids = inputs, 
    #     streamer = text_streamer, 
    #     max_new_tokens = max_seq_length, 
    #     use_cache = True,
    #     pad_token_id = tokenizer.eos_token_id,
    #     eos_token_id = tokenizer.eos_token_id,
    # )
    

# %%
result = run("오늘 우리반과 옆반의 평균 온도차이 알려줘")
# parsed_response_dict = eval(result)

# print(parsed_response_dict)


# %%

run("오늘 우리반 평균온도 알려줘")

run("어제 옆반 온도 평균 알려줘")
run("현재 우리반 실내온도 알려줘")

# %%
run("롯데캐슬의 현재 온도 알려줘")
run("10년전 우리반 온도 알려줘")

# %%
run("지금 옆반 에어컨 상태 알려줘")


