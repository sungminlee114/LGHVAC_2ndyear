__all__ = ['OperationExecutor']

import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np

class OperationExecutor:

    @classmethod
    def run_script(cls, args, scripts, returns):
        try:
            # logger.info(f'Executing operation {python_script}')
            # print(python_script, flush=True)
            # inject arguments into the global namespace
            globals().update(args)

            for script in scripts:
                try:
                    exec(script, globals())
                except Exception as e:
                    logger.error(f'Error executing operation {script}')
                    logger.error(e)
                    raise e
            # return variables named in the returns list
            return {name: globals()[name] for name in returns}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {name: None for name in returns}
    
    @classmethod
    def execute(cls, args, python_script, returns):
        result = cls.run_script(args, python_script, returns)
        for k, v in result.items():
            while True:
                if type(v) in [pd.DataFrame]:
                    # sort by timestamp
                    if "timestamp" in v.columns:
                        v = v.sort_values(by="timestamp")
                    
                    v['timestamp'] = v['timestamp'].map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
                    break
                
                # pd.Index
                elif type(v) in [pd.Index, np.ndarray, pd.Series]:
                    if len(v) == 0:
                        v = v.tolist()
                        continue

                    if type(v) in [pd.Series]:
                        v.reset_index(drop=True, inplace=True)

                    v = pd.unique(v)
                    v = pd.Series(v)
                    if type(v[0]) in [pd.Timestamp, datetime.date, datetime.datetime, np.datetime64]:
                        v = v.map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
                    v = v.tolist()
                    # remove -1 in the list
                    v = [x for x in v if x not in [-1, np.nan]]
                    if len(v) > 5:
                        v = v[:5]
                    break
                
                elif type(v) in [np.int64, np.float64, np.bool]:
                    v = v.item()
                elif type(v) in [np.datetime64]:
                    v = pd.Timestamp(v)
                elif type(v) in [pd.Timestamp, datetime.date, datetime.datetime]:
                    v = v.strftime("%Y-%m-%d %H:%M:%S")
                elif type(v) in [int, float, bool, str, list, dict]:
                    if type(v) in [int, float]:
                        if v in [-1, np.nan]:
                            v = None
                    elif type(v) in [list, dict]:
                        if len(v) == 0:
                            v = None
                    break
                else:
                    logger.error(f"Type not handled: {k}: {type(v), v}")
                    result[k] = None
                    break
            result[k] = v
        return result