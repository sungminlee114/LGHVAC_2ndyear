import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes
from src.input_to_instructions.load_and_execute import Semantic

class InstructionToSql:

    @classmethod
    def execute(cls, question: str, semantic: Semantic):
        prompt = """user

        Generate a SQL query to answer this question: `{question}`

        DDL statements:

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

        Prior information:
            {semantic}
        
        Prior information usage:
            Replace t[0] in question to t[1] for t in semantic['Temporal']
            Replace s[0] in question to s[1] for s in semantic['Spatial']
            Replace m[0] in question to m[1] for m in semantic['Modality']
        
        The following SQL query best answers the question `{question}`:
        ```sql
        """
        # semantic을 포함하여 prompt를 업데이트
        updated_prompt = prompt.format(question=question, semantic=semantic)
        inputs = cls.tokenizer(updated_prompt, return_tensors="pt").to("cuda")
        generated_ids = cls.model.generate(
            **inputs,
            num_return_sequences=1,
            eos_token_id=cls.tokenizer.eos_token_id,
            pad_token_id=cls.tokenizer.eos_token_id,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
        )
        outputs = cls.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return outputs[0].split("```sql")[1].split(";")[0]
  
  
if __name__ == "__main__":
    try:
        result = InstructionToSql.execute(
            "오늘 우리반과 옆반의 실내온도 차이 알려줘", '''{{
                    "Temporal": [("오늘", "2022-09-30 00:00:00 ~ 2022-09-30 23:59:59")],
                    "Spatial": [("우리반", "01_IB5"), ("옆반", "01_IB7")],
                    "Modality": ["실내온도"],
                    "Type/Quantity": ["diff"],
                    "Target": ["온도"],
                    "Question/Actuation": ["알려줘"]
            }}'''
        )
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")
