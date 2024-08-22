"""
The functions in this file is most likely to be used only once when creating the database for the first time.
"""

import logging

from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path

import numpy as np
import pandas as pd

from .instance import DBInstance

def initialize_db(db_instance: DBInstance):
    """
    This function is only used when creating the database for the first time.
    """
    # Optionally, delete the database if it already exists
    db_instance.delete_database('PerSite_DB')

    # Optionally, create a new database
    db_instance.create_database('PerSite_DB')

    # Switch to the desired database
    db_instance.switch_database('PerSite_DB')
    
    # Create a new table named 'idu_t'
    # It has id, name, and metadata columns
    # It will be used to store and manage the IDUs in other tables
    db_instance.create_table('idu_t', {
        'id': 'SERIAL PRIMARY KEY',
        'name': 'VARCHAR(50)',
        'metadata': 'VARCHAR(255)'
    })
    
    # Create a new hypertable named 'data_t'
    # It has id, idu_id, datas, and timestamp columns
    # Especially, the datas are: Roomtemp, Settemp, Oper
    
    db_instance.create_hypertable('data_t', {
        'id': 'SERIAL',
        'idu_id': 'INTEGER',
        'roomtemp': 'FLOAT',
        'settemp': 'FLOAT',
        'oper': 'BOOLEAN',
        'timestamp': 'TIMESTAMP NOT NULL'
    }, 'timestamp')

    # # Optionally, create a continuous aggregate
    # db_instance.create_continuous_aggregate('my_aggregate', "SELECT time_bucket('1 day', 'timestamp') AS day, AVG(value) FROM 'data_t' GROUP BY day")

    # # Optionally, set a retention policy
    # db_instance.set_retention_policy('data_t', '30 days')

def add_data_from_unzipped(db_instance: DBInstance, data_dir: Path):
    """
    This function is used to add data from an unzipped dir to the database.
    """
    
    if not data_dir.exists():
        logger.error(f"Directory {data_dir} does not exist")
        return

    db_instance.switch_database('PerSite_DB')
    
    for file in tqdm(sorted(data_dir.glob('*.npz'))[:7]):
        idu_name = "_".join(file.stem.split('_')[1:])
        db_instance.insert_data('idu_t', {'name': idu_name, 'metadata': 'Example metadata'}, ignore_if_exists=True)
        
        idu_id = db_instance.select_data('idu_t', ['id'], {'name': idu_name})[0]['id']
        logger.info(f"Inserting data for IDU {idu_name} with ID {idu_id}")
        
        if file.suffix == '.npz':
            data = np.load(file, allow_pickle=True)
            data_settemp = data['settemp'].tolist()
            data_roomtemp = data['roomtemp'].tolist()
            data_oper = data['oper'].astype(bool).tolist()
            data_timestamp = pd.to_datetime(data['timestamp'], unit='s')
        
            data_to_insert = [{
                'idu_id': idu_id,
                'roomtemp': roomtemp,
                'settemp': settemp,
                'oper': oper,
                'timestamp': timestamp
            } for roomtemp, settemp, oper, timestamp in \
                zip(data_roomtemp, data_settemp, data_oper, data_timestamp)]
            
        elif file.suffix == '.csv':
            df = pd.read_csv(file)
            
                    
            """
            df: 
                server_time  settemp  roomtemp  oper  pipe_outtemp  pipe_intemp   lev
            0  1.659312e+09     24.0      26.5   0.0          25.3         25.7  40.0
            1  1.659312e+09     24.0      26.5   0.0          25.3         25.7  40.0
            2  1.659312e+09     24.0      26.5   0.0          25.3         25.7  40.0
            3  1.659312e+09     24.0      26.5   0.0          25.3         25.7  40.0
            4  1.659312e+09     24.0      26.5   0.0          25.3         25.7  40.0
            
            convert this like below:
            data_to_insert = [
                {'idu_id': 1, 'roomtemp': 25.0, 'settemp': 22.0, 'oper': True, 'timestamp': '2022-01-01 00:00:00'},
                {'idu_id': 2, 'roomtemp': 26.0, 'settemp': 23.0, 'oper': False, 'timestamp': '2022-01-01 00:01:00'},
                {'idu_id': 1, 'roomtemp': 24.0, 'settemp': 21.0, 'oper': True, 'timestamp': '2022-01-01 00:02:00'},
                {'idu_id': 2, 'roomtemp': 27.0, 'settemp': 24.0, 'oper': False, 'timestamp': '2022-01-01 00:03:00'},
                {'idu_id': 1, 'roomtemp': 25.0, 'settemp': 22.0, 'oper': True, 'timestamp': '2022-01-01 00:04:00'},
                {'idu_id': 2, 'roomtemp': 26.0, 'settemp': 23.0, 'oper': False, 'timestamp': '2022-01-01 00:05:00'},
            ]
            """
            
            data_to_insert = df.assign(
                idu_id=idu_id,
                roomtemp=df['roomtemp'].astype(float),
                settemp=df['settemp'].astype(float),
                oper=df['oper'].astype(bool),
                timestamp=pd.to_datetime(df['server_time'], unit='s')
            ).drop(columns=['server_time', 'pipe_outtemp', 'pipe_intemp', 'lev']).to_dict('records')
        
        # Insert data into the database
        db_instance.insert_data('data_t', data_to_insert)
                    
    

if __name__ == "__main__":
    # When creating the database for the first time
    db_instance = DBInstance(dbname='postgres')
    # initialize_db(db_instance)
    add_data_from_unzipped(db_instance, Path('/dataset/LG/3_processed/0418_YongDongIllHigh_school/sr60/raw'))
    db_instance.close()