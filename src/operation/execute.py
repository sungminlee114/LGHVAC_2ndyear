import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np

class OperationExecutor:
    @classmethod
    def execute(cls, args, python_script, returns):
        # logger.info(f'Executing operation {python_script}')
        # print(python_script, flush=True)
        # inject arguments into the global namespace
        for name, value in args.items():
            globals()[name] = value
        
        scripts = python_script.split(';')
        for script in scripts:
            # print(script)
            # if script == "dates = daily_avg_temp[daily_avg_temp['settemp'] < daily_avg_temp['roomtemp']].index;":
            #     print(daily_avg_temp[daily_avg_temp['settemp'] < daily_avg_temp['roomtemp']].index)
            # print(script)
            try:
                exec(script, globals())
            except Exception as e:
                logger.error(f'Error executing operation {script}')
                logger.error(e)
                raise e
            # print([name for name in globals() if not name.startswith('_')])
        
        # return variables named in the returns list
        return {name: globals()[name] for name in returns}