import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes
from src.input_to_instructions.load_and_execute import Instruction, Semantic

class InstructionToSql:

    @classmethod
    def execute(cls, current_instruction:str, semantic: Semantic):
        question = current_instruction
        print(question)
          
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
   
            
        SQL Query Construction Linked to Semantic
        1. SELECT Clause:
        - Based on `semantic['Operation']`, identify whether to use aggregate functions (e.g., AVG, MAX, MIN) or simple column selections.
        - Specify columns from `semantic['Target']`, such as temperature differences, maximum temperatures, etc.

        2. FROM Clause:
        - Use `data_t` as the primary table. If necessary, join with `idu_t` or other relevant tables based on the context.

        3. WHERE Clause:
        - Apply filters using conditions from `semantic['Temporal']` to specify date ranges or timestamps.
        - Use subqueries or joins to match `idu_id` based on room names from `idu_t`, reflecting `semantic['Spatial']`.
        - Always incude NULL and 'NaN' values. Very important! But you don't have to include this on Boolean. Very important! If Modality is not "전원", include IS NOT NULL in sql query. 

        4. Check the SELECT value is not null. For this, include IS NOT NULL in sql query. But if the SELECT value is Boolean type, then you don't need to include the IS NOT NULL in sql query. 
        
        5. GROUP BY Clause
        - Include a GROUP BY clause if needed based on the aggregation or grouping requirements.

        6. ORDER BY Clause:
        - Include sorting criteria based on the result of aggregate functions when applicable. see the semantic['Modality']

        7. Handling Multiple Tables:
        - Clearly define how each table connects, using JOINs or subqueries to manage complex relationships.

        8. Only output the SQL query itself without any additional sentences or words. important!

        The following columns can be used as dt.column_name and dt.other_column in your SQL queries:
            id: The unique identifier for each record (integer).
            idu_id: The identifier for the associated room or unit (integer).
            roomtemp: The room temperature (double precision).
            settemp: The set temperature (double precision).
            oper: A boolean indicating the operational status (e.g., true or false).
            timestamp: The timestamp when the record was created (timestamp without time zone).

        Example :
       - question : 우리반에서 가장 더웠던 날이 언제야?
       - semantic : Semantic(Temporal=[('이번 달', '2022-09-01 00:00:00 ~ 2022-09-30 23:59:59')], Spatial=[('우리반', '01_IB5')], Modality=[('실내온도', 'roomtemp')], Operation=['최고'], Target=['날짜'])
       - Reasoning Process
           1. SELECT Clause: The SELECT clause is informed by the Target in the semantic, which specifies that we want to retrieve the date; thus, we select the date.
           2. FROM Clause: The FROM clause references the data_t table, which contains the relevant temperature data and aligns with the Modality indicating we are measuring room temperature.
           3. WHERE Clause: In the WHERE clause, we filter the results based on the timestamp range corresponding to the Temporal element, which defines "이번달" (this month). We also include a subquery to filter by idu_id, reflecting the Spatial element for the room "우리반" (01_IB5). Additionally, we ensure that roomtemp is neither NULL nor 'NaN' to maintain data integrity.
           4. Check the SELECT value is not null. For this, include IS NOT NULL in sql query. Since SELECT value is not Boolean type (dt."timestamp"), include the IS NOT NULL in sql query. 
           5. GROUP BY Clause: The GROUP BY clause is informed by the Target, which indicates we want to find the date. We will group the data by dt."timestamp" to aggregate temperature readings by each date.
           6. ORDER BY Clause: The ORDER BY clause utilizes the Operation element, which specifies "highest" (or similar terms). We sort the results by maximum temperature to identify the hottest day easily. Specifically, we can use MAX(dt.roomtemp) to determine the highest temperature for each date.
           7. LIMIT Clause: Finally, the LIMIT 1 clause ensures we return only the single hottest day, directly addressing the user's request for the maximum temperature in the specified timeframe and room.
        - Correct answer :  SELECT dt."timestamp"::date AS date
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
                            ORDER BY MAX(dt.roomtemp) DESC
                            LIMIT 1;
         
        Example : 
       - question : 현재 옆반 온도 알려줘
       - semantic : Semantic(Temporal=[('현재', '2022-09-30 12:00:00')], Spatial=['옆반'], Modality=['실내온도'], Operation=[None], Target=['온도'])
       - Reasoning Process
            1. SELECT Clause: The `SELECT` clause is informed by the `Target` in the semantic, which specifies that we want to retrieve the temperature (`온도`). Therefore, we select `dt.roomtemp` to fetch the room temperature data.
            2. FROM Clause: The `FROM` clause references the `data_t` table, which contains relevant temperature data. This aligns with the `Modality` indicating we are measuring indoor temperature (`실내온도`).
            3. WHERE Clause:
                - We filter results based on the timestamp defined by the `Temporal` element, specifically for the current time, which is set to `'2022-09-30 12:00:00'`.
                - We include a subquery to filter by `idu_id`, which corresponds to the spatial context of the room `옆반` (the adjacent room). The subquery selects the `id` from `idu_t` where the room name matches `옆반`.
                - Additionally, we ensure that `roomtemp` is not `NULL` and is distinct from `'NaN'` to maintain data integrity. This is done using the conditions `dt.roomtemp IS NOT NULL` and `dt.roomtemp IS DISTINCT FROM 'NaN'`.
            4. Check for Non-Null Values: It is essential to check that the `SELECT` value is not `NULL`. This is enforced in the SQL query by including `dt.roomtemp IS NOT NULL`.
            5. GROUP BY Clause: Since the `Target` indicates we want to find the temperature readings for specific timestamps, we do not need to group by date in this case.
            6. ORDER BY Clause: In this specific query, since we are targeting a single timestamp, we do not need to apply an ORDER BY clause. 
            7. LIMIT Clause: Finally, the `LIMIT 1` clause ensures we return only a single result, specifically the temperature reading for the given timestamp, thus addressing the user's request directly.

        - Correct answer :  SELECT dt.roomtemp FROM data_t dt WHERE dt."timestamp" = '2022-09-30 12:00:00' AND dt.idu_id = (SELECT id FROM idu_t WHERE name = '01_IB7') AND dt.roomtemp IS NOT NULL AND dt.roomtemp IS DISTINCT FROM 'NaN';
         
                            
        Example : 
        - question : 이번달 우리반의 온도 평균과 옆반의 온도평균의 차이 알려줘
        - semantic : Semantic(Temporal=[('이번달', '2022-09-01 00:00:00 ~ 2022-09-30 23:59:59')], Spatial=['우리반', '옆반'], Modality=['실내온도'], Operation=['평균', '차이'], Target=['온도'])
        - Reasoning Process
            1. SELECT Clause: The Target specifies "온도" (temperature), so we will select the difference of the average temperatures: AVG(dt1.roomtemp) - AVG(dt2.roomtemp).
            2. FROM Clause: We will join the data_t table on itself. We use dt1 for "우리반" and dt2 for "옆반" to compare their average temperatures, as we need to match timestamps for averaging.
            3. WHERE Clause:
            - The Temporal information specifies that we need records for the current month, so we set the timestamp range accordingly.
            - We filter dt1 for "우리반" (01_IB5) and dt2 for "옆반" (01_IB7) by matching the idu_id.
            - Both roomtemp values must be neither NULL nor 'NaN'. Since Modality is not "전원", include IS NOT NULL in sql query.
            4. Check the SELECT value is not null. For this, include IS NOT NULL in sql query. Since SELECT value is not Boolean type (AVG(dt1.roomtemp) - AVG(dt2.roomtemp)), include the IS NOT NULL in sql query. 
            5. GROUP BY Clause: We will group by the identifiers for both rooms to ensure we get the correct averages.
            6. ORDER BY Clause: Not necessary since we are interested in the difference between averages.
        - Correct Answer : 
            SELECT 
                AVG(dt1.roomtemp) - AVG(dt2.roomtemp) 
            FROM 
                data_t dt1
            JOIN 
                data_t dt2 ON dt1."timestamp" = dt2."timestamp"
            WHERE 
                dt1.idu_id = (SELECT id FROM idu_t WHERE name = '01_IB5') 
                AND dt2.idu_id = (SELECT id FROM idu_t WHERE name = '01_IB7') 
                AND dt1."timestamp" BETWEEN '2022-09-01 00:00:00' AND '2022-09-30 23:59:59' 
                AND dt1.roomtemp IS NOT NULL 
                AND dt1.roomtemp IS DISTINCT FROM 'NaN' 
                AND dt2.roomtemp IS NOT NULL 
                AND dt2.roomtemp IS DISTINCT FROM 'NaN'
            GROUP BY 
                dt1.idu_id, dt2.idu_id;
                           
        Example : 
        - question : 지금 옆반 에어컨 전원 꺼져있어?
        - semantic : Semantic(Temporal=[('지금', '2022-09-30 12:00:00')], Spatial=['옆반'], Modality=['전원'], Operation=[None], Target=['전원'])
        - Reasoning Process
            1. SELECT Clause: The SELECT clause is informed by the Target in the semantic, which specifies that we want to retrieve the operational status (전원). Therefore, we select dt.oper to fetch the operational data for the air conditioning system.
            2. FROM Clause: The FROM clause references the data_t table, which contains relevant operational data. This aligns with the Modality indicating we are measuring the system's operational status. 
            3. WHERE Clause:
                - We filter results based on the timestamp defined by the Temporal element, specifically for the current time, set to '2022-09-30 12:00:00'.
                - We filter dt2 for "옆반" (01_IB7) by matching the idu_id.
                - Since oper is a boolean value, it does not require a check for 'NaN'.
            4. Check for Non-Null Values: Since oper is a boolean value, there’s no need to check for NULL or 'NaN' in the SQL query.
            5. GROUP BY Clause: Not needed in this case, as we are targeting a specific timestamp.
            6. ORDER BY Clause: Not required since we are interested in a single result.
            7. LIMIT Clause: Finally, the `LIMIT 1` clause ensures we return only a single result, specifically the temperature reading for the given timestamp, thus addressing the user's request directly.
        - Correct Answer : SELECT dt.oper FROM data_t dt WHERE dt."timestamp" = '2022-09-30 12:00:00' AND dt.idu_id = (SELECT id FROM idu_t WHERE name = '01_IB7');
      


        
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
            temperature=0.00001,
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
