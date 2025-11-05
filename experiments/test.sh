../llama.cpp/build/bin/llama-cli -m ../src/i2i.gguf \
-n 1000 -c 1000 -st --threads 255 --temp 0.0 --top_p 1.0 --seed 42 -ngl 33 \
--system-prompt-file "prompt.temp" \
-p "Metadata:{'site_name': 'YongDongIllHighSchool', 'user_name': '홍길동', 'user_role': 'customer', 'idu_name': '01_IB5', 'idu_mapping': {'01_IB5': ['우리반'], '01_IB7': ['옆반'], '02_I81': ['앞반']}, 'modality_mapping': {'roomtemp': ['실내온도'], 'settemp': ['설정온도'], 'oper': ['전원']}, 'current_datetime': '2022-09-30 12:00:00'};Input:Why is our classroom so cold;" \
# --chat-template llama3