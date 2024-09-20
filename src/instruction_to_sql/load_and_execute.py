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
            
        SQL Query Construction Linked to Semantic
        1. SELECT Clause:
        - Connection to Semantic: Select columns based on `semantic['Target']`, specifying the values to retrieve, such as the date and the result of an aggregate function (e.g., AVG, MAX).

        2. FROM Clause:
        - Connection to Semantic: Data is sourced from the `data_t` table, which contains measurements relevant to the `Modality` (e.g., temperature).

        3. WHERE Clause:
        - Connection to Semantic:
            - Timestamp Condition: Filter records using conditions derived from `semantic['Temporal']`, such as a specific date or a date range.
            - Room ID Filter: Use a subquery to select the `idu_id` based on the specific name from `idu_t`, linked to `semantic['Spatial']`.
            - Data Integrity: It is very important. Ensure the selected column is not NULL and not 'NaN'. But if selected column is boolean type, then do not include this.

        4. GROUP BY Clause:
        - Connection to Semantic: Group results based on `semantic['Type_Quantity']`, which indicates whether to perform a statistical operation like avg, argmax.

        5. ORDER BY Clause:
        - Connection to Semantic: If `semantic['Type_Quantity']` indicates a statistical operation (e.g., AVG), order the results based on `semantic['Modality']` and the chosen aggregate function.

        6. Flexibility in Column Selection:
        - Connection to Semantic: `dt.column_name` and `dt.other_column` can be the same or different, allowing for varied analysis and presentation of dat
        
        Following columns can be used as dt.column_name and dt.other_column in your SQL queries. 
            id: The unique identifier for each record (integer).
            idu_id: The identifier for the associated room or unit (integer).
            roomtemp: The room temperature (double precision).
            settemp: The set temperature (double precision).
            oper: A boolean indicating the operational status (e.g., true or false).  false means "꺼짐", true means "켜짐".
            timestamp: The timestamp when the record was created (timestamp without time zone).

        - Correct answer:  
                            '''sql
                            SELECT dt.column_name, AGG_FUNC(dt.other_column) AS aggregate_value
                            FROM data_t dt
                            WHERE (dt."timestamp" = 'specific_timestamp' OR dt."timestamp" BETWEEN 'start_date' AND 'end_date')
                            AND dt.idu_id IN (
                                SELECT id
                                FROM idu_t
                                WHERE name = 'specific_name'
                            )
                            AND dt.column_name IS NOT NULL
                            AND dt.column_name IS DISTINCT FROM 'NaN'
                            GROUP BY dt.idu_id
                            ORDER BY aggregate_value [ASC|DESC];  -- This allows you to specify the order as needed.         
       Example :
       - question : 우리반에서 가장 더웠던 날이 언제야?
       - semantic : Semantic(Temporal=[('이번 달', '2022-09-01 00:00:00 ~ 2022-09-30 23:59:59')], Spatial=[('우리반', '01_IB5')], Modality=[('실내온도', 'roomtemp')], Type_Quantity=['argmax'], Target=['날짜'], Question_Actuation=['알려줘'])
       - Reasoning Process
           1. SELECT Clause: The SELECT clause is informed by the Target in the semantic, which specifies that we want to retrieve the date and the maximum temperature; thus, we select the date and MAX(dt.roomtemp).
           2. FROM Clause: The FROM clause references the data_t table, which contains the relevant temperature data and aligns with the Modality indicating we are measuring room temperature.
           3. WHERE Clause: In the WHERE clause, we filter the results based on the timestamp range corresponding to the Temporal element, which defines "이번달" (this month). We also include a subquery to filter by idu_id, reflecting the Spatial element for the room "우리반" (01_IB5). Additionally, we ensure that roomtemp is neither NULL nor 'NaN' to maintain data integrity.
           4. GROUP BY Clause: The GROUP BY clause is used to aggregate temperatures by date, aligning with the Type_Quantity that indicates we want to find the maximum value (argmax).
           5. ORDER BY Clause: The ORDER BY clause sorts the results by maximum temperature, making it easy to identify the hottest day.
           6. LIMIT Clause: Finally, the LIMIT 1 clause ensures we return only the single hottest day, directly addressing the user's request for the maximum temperature in the specified timeframe and room.
        - Correct answer :  SELECT dt."timestamp"::date AS date, MAX(dt.roomtemp) AS max_temp
                            FROM data_t dt
                            WHERE dt."timestamp" BETWEEN '2022-09-01 00:00:00' AND '2022-09-30 23:59:59'
                            AND dt.idu_id IN (
                                SELECT id
                                FROM idu_t
                                WHERE name = '01_IB5'
                            )
                            AND dt.roomtemp IS NOT NULL
                            AND dt.roomtemp IS DISTINCT FROM 'NaN'
                            GROUP BY date
                            ORDER BY max_temp DESC
                            LIMIT 1;

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
