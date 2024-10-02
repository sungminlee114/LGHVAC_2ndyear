import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import queue
import time
import pprint
import concurrent.futures

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask import stream_with_context

from src.core import *


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 모든 도메인에서의 요청을 허용

@app.route('/process', methods=['GET']) 
def process_request():
    sentence = request.args.get('sentence')
    logger.info(f"Processing request: {sentence}")
    if not sentence:
        return jsonify({'error': 'Sentence query parameter is required!'}), 400
    try:
        # Using a timeout to ensure it doesn't hang for too long
        def response_function(response):
            response = response.replace("\n", "<br>")
            logger.info(f"A response is added: {response}")
            yield f"data: {response}\n\n"
        
        def stream():
            current_metadata = get_current_metadata()
            # response_function("<h1 Current metadata>")
            # response_function(pprint.pformat(current_metadata))
            
            semantic, instruction_set = input_to_instruction_set(sentence, current_metadata)
            yield from response_function("<h1 Input to instruction result>")
            yield from response_function("<h2 Semantic:>")
            yield from response_function(semantic.pformat())
            yield from response_function("<h2 Instruction set:>")
            yield from response_function(pprint.pformat(instruction_set))
            
            # Call the instruction set execution
            yield from execute_instruction_set_web(semantic, instruction_set, sentence, current_metadata, response_function)
            
            # Once processing is complete, we put a 'finish' flag
            yield from response_function("finish")
            
        
        # Return results as an event stream
        return Response(
            stream(), 
            mimetype='text/event-stream',
        )
    
    except concurrent.futures.TimeoutError:
        return jsonify({'error': 'Processing took too long, please try again.'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500

load_models()
import subprocess
subprocess.Popen(['python', '-m', 'http.server', '9010'])

if __name__ == "__main__":
    # 개발용으로만 Flask 내장 서버를 실행하고 싶다면 아래 코드를 사용하세요.
    app.run(host='0.0.0.0', port=9011, debug=False, threaded=True, use_reloader=False)