import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes

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
    def query_id(cls, question: str):
        # Extract the location from the question
        # Assuming the question is something like "오늘 [장소] 방온도 알려줘"
        location = cls.extract_location(question)
        
        # Create a new prompt to find the ID using the extracted location
        prompt = f"""user

        Generate a SQL query to find the ID associated with the location "{location}". 

        DDL statements:

        CREATE TABLE idu_t (
        id INTEGER PRIMARY KEY, -- ID for IDU
        name VARCHAR(50) PRIMARY KEY, -- name for IDU 
        metadata VARCHAR(255) PRIMARY KEY, -- metadata for IDU
        );

        The following SQL query best finds the ID for "{location}":
        ```sql
        SELECT id FROM idu_t WHERE name = '{location}';
        ```
        """
        sql_id_string = cls.execute_module(question, prompt)
        return sql_id_string
    
    @classmethod
    def query_sql(cls, question: str, id: int):
        # Update the prompt to generate a SQL query that answers the question based on the given ID
        prompt = f"""user

        Generate a SQL query to answer the following question: `{question}`

        DDL statements:

        CREATE TABLE idu_t (
        id INTEGER PRIMARY KEY, -- ID for IDU
        idu_id INTEGER PRIMARY KEY, -- idu id for IDU 
        roomtemp DOUBLE PRIMARY KEY, -- room temperature for IDU
        settemp DOUBLE PRIMARY KEY, -- set temperature for IDU
        oper BOOL PRIMARY KEY, -- operation for IDU
        timestamp PRIMARY KEY, -- timestamp for IDU
        );

        The following SQL query should answer the question `{question}` based on the provided ID:
        ```sql
        SELECT * FROM idu_t WHERE id = {id};
        ```
        """
        result_sql_string = cls.execute_module(question, prompt)
        return result_sql_string
    
    @classmethod
    def execute_module(cls, question: str, prompt: str):
        if not question:
            raise ValueError("The question should not be empty.")
        
        updated_prompt = prompt.format(question=question)
        inputs = cls.tokenizer(updated_prompt, return_tensors="pt").to("cuda")
        generated_ids = cls.model.generate(
            **inputs,
            num_return_sequences=1,
            eos_token_id=cls.tokenizer.eos_token_id,
            pad_token_id=cls.tokenizer.eos_token_id,
            max_new_tokens=400,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            top_p=1,
        )
        outputs = cls.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Ensure that the SQL query extraction is robust
        try:
            sql_output = outputs[0].split("```sql")[1].split("```")[0].strip()
        except IndexError:
            raise ValueError("The SQL query could not be extracted from the model output.")
        
        return sql_output
    
    @classmethod
    def extract_location(cls, question: str):
        # Extract the location from the question
        # Assumes the location is between "오늘" and "방온도" in the question
        # Adjust the extraction logic based on the actual format of the question
        parts = question.split(" ")
        if len(parts) >= 2:
            return parts[1]  # Assuming the location is the second part
        raise ValueError("Could not extract location from the question.")
    
    @classmethod
    def extract_id(cls, query_result: str):
        # Extract the ID from the query result string
        # Example result: "SELECT id FROM idu_t WHERE name = 'IO';"
        try:
            # Example extraction assuming the ID is just a number
            # You might need to adjust this based on actual output format
            lines = query_result.split('\n')
            for line in lines:
                if 'SELECT id FROM idu_t WHERE name = \'IO\';' in line:
                    return line.split('SELECT id FROM idu_t WHERE name = \'IO\';')[1].strip()
            raise ValueError("The ID could not be extracted from the query result.")
        except Exception as e:
            raise ValueError(f"Error extracting ID: {e}")
    
    @classmethod
    def example(cls, question: str):
        # Get the ID first
        id_query = cls.query_id(question)
        # Extract the ID from the query result
        id = cls.extract_id(id_query)
        
        # Use the ID in the SQL query to get the information
        result_sql_string = cls.query_sql(question, int(id))
        
        return result_sql_string

if __name__ == "__main__":
    try:
        result = InstructionToSql.example(
            "오늘 우리집 방온도 알려줘"
        )
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")
