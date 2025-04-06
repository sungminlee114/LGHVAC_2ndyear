import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pprint
import time
import concurrent.futures

from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS

from src.core import *
import src.demo_helper as demo_helper
import src.status_util as status_util

available_metadatas:dict = demo_helper.get_available_metadatas()

app = Flask(
    __name__, 
    template_folder='web',
    static_folder='web'
)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

CORS(app, resources={r"/*": {"origins": "*"}})  # 모든 도메인에서의 요청을 허용

port_number = 9014

@app.route('/')
def index():
    html_content = render_template('demo.html', port_number=port_number)
    return html_content

@app.route('/process', methods=['GET']) 
def process_request():
    start_time = time.time()  # 시작 시간 기록
    
    sentence = request.args.get('sentence')
    metadata_name = request.args.get('metadata_name')
    logger.info(f"Processing request: {sentence}, {metadata_name}")
    if not sentence or not metadata_name:
        return jsonify({'error': 'Sentence and metadata are required.'}), 400

    try:
        # Using a timeout to ensure it doesn't hang for too long
        def response_function(response, message_type="debug"):
            # 메시지 타입을 추가하여 클라이언트에게 전송
                
            response = response.replace("\n", "<br>")
            if message_type in ["response"]:
                logger.info(f"A response is added: {response}")
            # 메시지 타입을 포함하여 전송
            yield f"data: {json.dumps({'type': message_type, 'content': response})}\n\n"

        def stream():
            current_metadata = dict(available_metadatas[metadata_name]["metadata"])
            
            # 디버그 메시지 시작
            # yield from response_function("디버그 메시지 시작")
            
            instructions = input_to_instruction_set(sentence, current_metadata)
            yield from response_function(f"Instructions: {pprint.pformat(instructions)}")

            yield from execute_instruction_set_web(instructions, sentence, current_metadata, response_function)
            
            # 처리 완료 후 시간 측정 및 로깅
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            # 최종 응답 후 처리 시간 알림
            yield from response_function(f"{processing_time:.2f}", "time")
            
            # 처리 완료 신호
            yield from response_function("", "finish")
            
        # Return results as an event stream
        return Response(
            stream(), 
            mimetype='text/event-stream',
        )
    
    except concurrent.futures.TimeoutError:
        end_time = time.time()
        logger.error(f"Timeout error after {end_time - start_time:.2f} seconds")
        return jsonify({'error': 'Processing took too long, please try again.'}), 504
    except Exception as e:
        end_time = time.time()
        logger.error(f"Error occurred after {end_time - start_time:.2f} seconds: {str(e)}")
        load_models()
        return jsonify({'error': str(e)}), 500

@app.route('/available_metadatas', methods=['GET'])
def get_metadata():
    """Returns available metadatas.

    Returns:
        available_metadatas (json): A json object containing available metadatas structured as follows:
            {"scenario_name": {"metadata_key": "metadata_value", ...}, ...}
        
    """
    
    metadatas = jsonify(available_metadatas)
    return metadatas

@app.route('/status', methods=['GET']) 
def get_status():
    vram_stats = status_util.get_gpu_memory_nvml()
    model_status = status_util.check_model_load_status()
    status = {
        'model_status': model_status,
        'vram_status': vram_stats
    }
    return jsonify(status)

load_models()
if __name__ == "__main__":
    # 개발용으로만 Flask 내장 서버를 실행하고 싶다면 아래 코드를 사용하세요.
    
    app.run(host='0.0.0.0', port=port_number, debug=True, threaded=True, use_reloader=False)