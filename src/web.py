import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pprint
import concurrent.futures
import io
import base64

import matplotlib.figure

from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS

from src.core import *
import src.demo_helper as demo_helper

available_metadatas:dict = demo_helper.get_available_metadatas()
print(available_metadatas)

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
    sentence = request.args.get('sentence')
    metadata_name = request.args.get('metadata_name')
    logger.info(f"Processing request: {sentence}, {metadata_name}")
    if not sentence or not metadata_name:
        return jsonify({'error': 'Sentence and metadata are required.'}), 400

    try:
        # Using a timeout to ensure it doesn't hang for too long
        def response_function(response, response_type=None):
            # if isinstance(response, matplotlib.figure.Figure):
            if response_type == "graph":
                buf = io.BytesIO()
                response.savefig(buf, format='png')
                buf.seek(0)
                response = base64.b64encode(buf.read()).decode('utf-8')
                response = f'<img src="data:image/png;base64,{response}"/>'
            else:
                response = response.replace("\n", "<br>")
                
            logger.debug(f"A response is added: {response}")
            yield f"data: {response}\n\n"

        def stream():
            current_metadata = dict(available_metadatas[metadata_name])
            # response_function("<h1 Current metadata>")
            # response_function(pprint.pformat(current_metadata))
            
            instructions = input_to_instruction_set(sentence, current_metadata)
            # yield from response_function("<h1 Input to instruction result>")
            # # yield from response_function("<h2 Semantic:>")
            # # yield from response_function(semantic.pformat())
            # yield from response_function("<h2 Instruction set:>")
            # yield from response_function(pprint.pformat(instructions))
            
            # Call the instruction set execution
            yield from execute_instruction_set_web(instructions, sentence, current_metadata, response_function)
            
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
        load_models()
        return jsonify({'error': str(e)}), 500

@app.route('/available_metadatas', methods=['GET'])
def get_metadata():
    """Returns available metadatas.

    Returns:
        available_metadatas (json): A json object containing available metadatas structured as follows:
            {"scenario_name": {"metadata_key": "metadata_value", ...}, ...}
        
    """

    metadatas = {}
    for k, v in available_metadatas.items():
        metadatas[k] = [
            ['Site 정보',
                [
                    # ["Site 이름", v.get('site_name', None)],
                    ["Modality 매핑", [f"{k}: {v}" for k, v in v.get('modality_mapping', {}).items()]],
                ]
            ],
            ['유저 정보', [
                ["이름", v.get('user_name', None)],
                # ["User 역할", v.get('user_role', None)],
                ["IDU 이름", v.get('idu_name', None)],
                ["IDU 매핑", [f"{k}: {v}" for k, v in v.get('idu_mapping', {}).items()]],
            ]],
            ['현재 정보', [
                ["일시", v.get('current_datetime', None),] 
            ]],
        ]

        # for kk, vv in v.items():
        #     if kk in ['site_name', 'modality_mapping']:
        #         metadatas[k]['site'][kk] = vv
        #     elif kk in ['user_name', 'user_role', 'idu_name', 'idu_mapping']:
        #         metadatas[k]['user'][kk] = vv
        #     else:
        #         metadatas[k]['current'][kk] = vv

    
    metadatas = jsonify(metadatas)
    return metadatas

load_models()
if __name__ == "__main__":
    # 개발용으로만 Flask 내장 서버를 실행하고 싶다면 아래 코드를 사용하세요.
    
    app.run(host='0.0.0.0', port=port_number, debug=True, threaded=False, use_reloader=False)