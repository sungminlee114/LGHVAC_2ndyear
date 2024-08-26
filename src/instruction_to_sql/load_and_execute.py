import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes
from src.input_to_instructions.load_and_execute import Semantic

class InstructionToSql:
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

            사전 정보는 다음과 같아.
            "Semantic": {semantic}
            'temporal'의 두 번째 인자(타임스탬프 형식)가 timestamp conditioning에 사용돼.
            spatial 의 두번째 인자가 idu_t.name 이야. 예를 들어, 01_IB5.
            modalitry의 두번째 인자가 data_t의 column이야. 예를들어 roomtemp.

        Example:
            Question: 오늘 우리반의 실내온도 알려줘.
            Semantic: Semantic(
                Temporal=[("오늘", "2022-09-30 00:00:00 ~ 2022-09-30 23:59:59")],
                Spatial=[('우리반', '01_IB5')]
                Modality=[("실내온도", 'roomtemp')],
                Type_Quantity=["value"],
                Target=["온도"]
                Question_Actuation=["알려줘"]
            )
            Output: ```sql
                SELECT data_t.roomtemp
                FROM data_t
                JOIN idu_t ON data_t.idu_id = idu_t.id
                WHERE idu_t.name = '01_IB5' AND data_t.timestamp >= '2022-09-30 00:00:00' AND data_t.timestamp <= '2022-09-30 23:59:59';
            ```


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
