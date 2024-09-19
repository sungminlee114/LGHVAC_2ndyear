import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import queue
import time
import concurrent.futures

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from src.core import *


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 모든 도메인에서의 요청을 허용

def process_sentence(sentence, result_queue:queue):
    current_metadata = get_current_metadata()
    
    semantic, instruction_set = input_to_instruction_set(sentence, current_metadata)
    
    def response_function(response):
        result_queue.put(response)
    
    # Call the instruction set execution
    execute_instruction_set(semantic, instruction_set, sentence, current_metadata, response_function)
    
    # Once processing is complete, we put a 'finish' flag
    result_queue.put("finish")

@app.route('/process', methods=['GET']) 
def process_request():
    sentence = request.args.get('sentence')
    print("Processing request", sentence)
    if not sentence:
        return jsonify({'error': 'Sentence query parameter is required!'}), 400

    result_queue = queue.Queue()

    try:
        # Using a timeout to ensure it doesn't hang for too long
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Process the sentence in a separate thread
            future = executor.submit(process_sentence, sentence, result_queue)
            
            def stream():
                while True:
                    try:
                        # Wait for new results from the queue
                        result = result_queue.get(timeout=10)  # 10 seconds timeout for results
                        print("Got result", result)
                        yield f"data: {result}\n\n"
                        if result == "finish":
                            break
                    except queue.Empty:
                        yield "data: [Error] No response available in time.\n\n"
                        break

            # Return results as an event stream
            return Response(stream(), mimetype='text/event-stream')
    
    except concurrent.futures.TimeoutError:
        return jsonify({'error': 'Processing took too long, please try again.'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def server_main():
    load_models()
    
    # open a server to serve demo.html
    import subprocess
    subprocess.Popen(['python', '-m', 'http.server', '8001'])
    
    app.run(host='0.0.0.0', port=8002, debug=True, use_reloader=False)

if __name__ == "__main__":
    server_main()