import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes
import sqlparse

def generate_query(question):
    
    prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{question}`

DDL statements:

    CREATE TABLE data_t (
        id INTEGER PRIMARY KEY, -- ID for IDU
        idu_id INTEGER PRIMARY KEY, -- idu id for IDU 
        roomtemp DOUBLE PRIMARY KEY, -- room temperature for IDU
        settemp DOUBLE PRIMARY KEY, -- set temperature for IDU
        oper BOOL PRIMARY KEY, -- operation for IDU
        timestamp PRIMARY KEY, -- timestamp for IDU
        );
        
    CREATE TABLE idu_t (
        id INTEGER PRIMARY KEY, -- ID for IDU
        name VARCHAR(50) PRIMARY KEY, -- name for IDU 
        metadata VARCHAR(255) PRIMARY KEY, -- metadata for IDU
        );

    사전 정보는 다음과 같아.
    "Semantic Parsing": {{
            "Temporal": [("오늘", "2022-09-30 00:00:00 ~ 2022-09-30 23:59:59")],
            "Spatial": [("우리반", "01_IB5"), ("옆반", "01_IB7")],
            "Modality": ["실내온도"],
            "Type/Quantity": ["diff"],
            "Target": ["온도"],
            "Question/Actuation": ["알려줘"]
    }}
    spatial 의 두번째 인자가 idu_t.name 이야. 예를 들어, 01_IB5.

The following SQL query best answers the question `{question}`:
```sql
"""
    
  
    updated_prompt = prompt.format(question=question)
    inputs = tokenizer(updated_prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
        temperature=0.0,
        top_p=1,
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return outputs[0].split("```sql")[1].split(";")[0]

if __name__ == "__main__":
    model_name = "defog/llama-3-sqlcoder-8b"
    tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="/model"
        )

    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="/model"
        )
        
    question = "오늘 우리반과 옆반의 실내온도 차이 알려줘"
    generated_sql = generate_query(question)
    print(sqlparse.format(generated_sql, reindent=True))
# 오늘 우리반의 실내온도 알려줘
#오늘 우리반과 옆반의 실내온도 차이 알려줘"
