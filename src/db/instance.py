import logging

logger = logging.getLogger(__name__)

import re
from copy import deepcopy

import pandas as pd
import psycopg2
from psycopg2 import sql

def parse_temporal(temporal):
    if temporal is None:
        return ""

    temporal = temporal.strip()
    if temporal[0] not in "[(" or temporal[-1] not in "])":
        return f"timestamp = {temporal}"
    else:
        # 외부 괄호 제거
        left_bracket = temporal[0]
        right_bracket = temporal[-1]
        inner = temporal[1:-1]

        # 중첩된 괄호를 고려하여 외부 콤마 위치 찾기
        paren_count = 0
        split_index = None
        for i, ch in enumerate(inner):
            if ch == '(':
                paren_count += 1
            elif ch == ')':
                paren_count -= 1
            elif ch == ',' and paren_count == 0:
                split_index = i
                break

        if split_index is None:
            raise ValueError(f"Invalid temporal format: {temporal}")

        start = inner[:split_index].strip()
        end = inner[split_index+1:].strip()

        conditions = []

        # 시작 시간 처리
        if start != '~':
            operator = ">=" if left_bracket == "[" else ">"
            conditions.append(f"timestamp {operator} {start}")

        # 종료 시간 처리
        if end != '~':
            operator = "<=" if right_bracket == "]" else "<"
            conditions.append(f"timestamp {operator} {end}")

        if not conditions:
            return None

        return " AND ".join(conditions)

class DBInstance:
    """
    This is a low-level DB driver class that provides methods to interact with a PostgreSQL database.
    """
    def __init__(self, host='localhost', port=5432, dbname='postgres', user='postgres', password='postgres'):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.connection = self.connect_to_db(dbname)
        self.cursor = self.connection.cursor()
        logger.info(f"Connected to the database {dbname}")

    @property
    def is_connected(self):
        return not self.connection.closed

    def connect_to_db(self, dbname):
        connection = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=dbname,
            user=self.user,
            password=self.password
        )
        connection.autocommit = True
        return connection

    def switch_database(self, dbname):
        self.cursor.close()
        self.connection.close()
        self.connection = self.connect_to_db(dbname)
        self.cursor = self.connection.cursor()
        logger.info(f"Switched to database {dbname}")

    def create_database(self, db_name):
        try:
            self.cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
            logger.info(f"Database {db_name} created successfully")
        except psycopg2.errors.DuplicateDatabase:
            logger.warning(f"Database {db_name} already exists")

    def delete_database(self, db_name):
        # Check if the database exists
        self.cursor.execute(
            sql.SQL("SELECT 1 FROM pg_database WHERE datname = {}").format(sql.Literal(db_name))
        )
        
        if not self.cursor.fetchone():
            logger.warning(f"Database {db_name} does not exist while trying to delete")
            return
        
        try:
            self.cursor.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(db_name)))
            logger.info(f"Database {db_name} deleted successfully")
        except Exception as e:
            logger.error(f"An error occurred while deleting database {db_name}: {e}")
        
        self.switch_database('postgres')
    
    def create_table(self, table_name, schema):
        """
        Creates a regular table in the database.
        
        Parameters:
        - table_name (str): The name of the table to be created.
        - schema (dict): A dictionary where the keys are column names and values are their data types.
        """
        try:
            self.cursor.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
                sql.Identifier(table_name),
                sql.SQL(', ').join(sql.SQL('{} {}').format(sql.Identifier(col_name), sql.SQL(data_type)) for col_name, data_type in schema.items())
            ))
            logger.info(f"Table {table_name} created successfully")
        except Exception as e:
            logger.error(f"An error occurred while creating table {table_name}: {e}")

    def create_hypertable(self, table_name, schema, time_column):
        """
        Creates a hypertable for TimescaleDB.
        
        Parameters:
        - table_name (str): The name of the table to be created.
        - schema (dict): A dictionary where the keys are column names and values are their data types.
        - time_column (str): The name of the column to be used as the time column for the hypertable.
        """
        # Reuse the create_table function to create the base table
        self.create_table(table_name, schema)
        
        try:
            # Convert the table into a hypertable
            self.cursor.execute(
                sql.SQL("SELECT create_hypertable(%s, %s, if_not_exists => TRUE)"),
                [table_name, time_column]
            )
            logger.info(f"Table {table_name} converted to a hypertable successfully")
        except Exception as e:
            logger.error(f"An error occurred while creating hypertable {table_name}: {e}")

    def execute_sql(self, sql:str) -> list[tuple] | None:
        """
        Executes an SQL query against the database.

        This method automatically determines whether to fetch results based on the query type.
        It fetches and returns results for SELECT queries, and commits changes for other queries
        like INSERT, UPDATE, DELETE, etc.

        Parameters:
        - sql (str): The SQL query to be executed.

        Returns:
        - list: A list of tuples containing the results if the query is a SELECT query.
        - None: If the query is not a SELECT query, or if an error occurs.

        Logs:
        - Info: When the query is executed successfully.
        - Error: If an exception occurs during query execution.
        """
    
        try:
            # Execute the SQL query
            self.cursor.execute(sql)
            
            # Automatically determine if the query is a SELECT
            if sql.strip().lower().startswith('select'):
                results = self.cursor.fetchall()
                logger.info("SQL SELECT query executed successfully and results fetched")
                return results
            else:
                # Commit the transaction for non-SELECT queries
                self.connection.commit()
                logger.info("SQL query executed successfully")
                return None
        except Exception as e:
            logger.error(f"An error occurred while executing SQL query: {e}")
            return None

    def insert_data(self, table_name, data, ignore_if_exists=False):
        """_summary_

        Args:
            table_name (str): 
            data (list[dict]):
            ignore_if_exists (bool, optional): _description_. Defaults to False.
        """
        try:
            if not isinstance(data, list):
                data = [data]  # Ensure data is a list of dictionaries

            columns = data[0].keys()
            insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                sql.Identifier(table_name),
                sql.SQL(', ').join(map(sql.Identifier, columns)),
                sql.SQL(', ').join(sql.Placeholder() * len(columns))
            )

            if ignore_if_exists:
                insert_query += sql.SQL(" ON CONFLICT DO NOTHING")
            
            # Execute the insert with executemany for multiple rows
            self.cursor.executemany(insert_query, [tuple(row.values()) for row in data])
            self.connection.commit()
            logger.info(f"Data inserted into {table_name} successfully")
        except Exception as e:
            logger.error(f"An error occurred while inserting data into {table_name}: {e}")

    def select_data(self, table_name, columns=None, condition=None):
        try:
            select_query = sql.SQL("SELECT {} FROM {}").format(
                sql.SQL(', ').join(map(sql.Identifier, columns)) if columns else sql.SQL('*'),
                sql.Identifier(table_name)
            )
            if condition:
                condition_sql = sql.SQL(' AND ').join(
                    sql.SQL("{} = {}").format(sql.Identifier(k), sql.Placeholder()) for k in condition.keys()
                )
                select_query += sql.SQL(" WHERE {}").format(condition_sql)

            self.cursor.execute(select_query, tuple(condition.values()) if condition else None)
            rows = self.cursor.fetchall()
            logger.info(f"Data selected from {table_name} successfully")
            
            # Convert the results to a list of dictionaries
            colnames = [desc[0] for desc in self.cursor.description]
            return [dict(zip(colnames, row)) for row in rows]
        except Exception as e:
            logger.error(f"An error occurred while selecting data from {table_name}: {e}")
            return None

    def execute_structured_query_string(self, query_string:str) -> pd.DataFrame:
        query_composed = sql.SQL(query_string)
        self.cursor.execute(query_composed)
        rows = self.cursor.fetchall()
        colnames = [desc[0] for desc in self.cursor.description]
        return pd.DataFrame(rows, columns=colnames)

    def structured_query_to_query_string(self, table_name, columns=None, conditions=None, subquery=None, get_rowids=False):
        """
        Select data from the database table with the option for subqueries and conditions.
        
        Args:
            table_name (str): Name of the table to query.
            columns (list, optional): List of columns to include in the SELECT statement.
            conditions (list, optional): List of raw SQL conditions to include in the WHERE clause.
            subquery (str, optional): Raw SQL for a subquery to include in the WHERE clause.
            
        Returns:
            df (pd.DataFrame): A DataFrame containing the selected data.
            
        Example:
            result = structured_query(
                table_name='data_t',
                columns=['idu_id', 'roomtemp'],
                conditions=[
                    "timestamp = '2022-09-30 00:00:00'",
                    "roomtemp IS NOT NULL",
                    "roomtemp IS DISTINCT FROM 'NaN'"
                ],
                subquery="idu_id = (SELECT id FROM idu_t WHERE name = '01_IB5')"
            )
            print(result) # DataFrame containing the selected data
        """
        try:
            # columns 리스트에서 각 항목이 문자열이면, 만약 "raw:" 접두어가 있다면 raw SQL로 처리하고,
            # 그렇지 않으면 sql.Identifier로 처리하도록 함.
            
            if get_rowids:
                if columns and "id" not in columns:
                    columns.append("id")
                else:
                    columns = ["id"]
            
            if "timestamp" not in columns:
                columns.append("timestamp")
            
            if columns:
                formatted_columns = []
                for col in columns:
                    if isinstance(col, str) and col.startswith("raw:"):
                        # "raw:" 뒤의 부분을 raw SQL로 감쌈
                        formatted_columns.append(sql.SQL(col[4:]))
                    else:
                        formatted_columns.append(sql.Identifier(col))
                
                select_columns = sql.SQL(', ').join(formatted_columns)
            else:
                select_columns = sql.SQL('*')
            
            select_query = sql.SQL("SELECT {} FROM {}").format(
                select_columns,
                sql.Identifier(table_name)
            )
            
            # Initialize a list for WHERE clause components
            where_clauses = []

            # Add subquery if provided
            if subquery and subquery != "~":
                where_clauses.append(subquery)
            
            # Add raw conditions if provided
            if conditions:
                where_clauses.extend(conditions)
            
            # for each column: check NULL and NaN
            for col in columns:
                # 만약 col이 "raw:" 접두어가 붙은 문자열이라면, 여기서는 해당 컬럼에 대한 NULL 체크는 생략할 수 있음.
                if isinstance(col, str) and col.startswith("raw:"):
                    continue
                if col == "timestamp":
                    continue
                
                where_clauses.append(sql.SQL("{} IS NOT NULL").format(sql.Identifier(col)))
                
                if col in ["roomtemp", "settemp"]:
                    where_clauses.append(sql.SQL("{} IS DISTINCT FROM 'NaN'").format(sql.Identifier(col)))
            
            # Add WHERE clause if there are any conditions or subqueries
            if where_clauses:
                where_part = " AND ".join([str(clause) if not isinstance(clause, sql.Composed) else clause.as_string(self.connection) for clause in where_clauses])
                select_query += sql.SQL(" WHERE {}").format(sql.SQL(where_part))
            
            # Add sorting by timestamp
            select_query += sql.SQL(" ORDER BY timestamp")
            
            logger.debug(f"Select query as string: {select_query.as_string(self.connection)}")
            return select_query.as_string(self.connection)
            
            # print(select_query.as_string(self.connection), flush=True)
            # Execute the query
            self.cursor.execute(select_query)
            rows = self.cursor.fetchall()
            logger.debug("Data selected from {} successfully".format(table_name))
            
            # Convert the results to a list of dictionaries
            colnames = [desc[0] for desc in self.cursor.description]
            # return [dict(zip(colnames, row)) for row in rows]
            result_dict = {col: [] for col in colnames}
            for row in rows:
                for col, value in zip(colnames, row):
                    result_dict[col].append(value)
            df = pd.DataFrame(result_dict)
            return df
            # return result_dict
        except Exception as e:
            logger.error("An error occurred while selecting data with\n{}".format(select_query.as_string(self.connection)))
            logger.error("Error message: {}".format(e))
            return None
    
    def get_query_strings(self, metadata, columns, temporal=None, spatials=None, get_rowids=False, exp_tag=None):
        """
        Returns a list of query strings for the given metadata, columns, temporal, spatials, and get_rowids.
        """
        temporal = parse_temporal(temporal)

        if temporal != None:
            current_datetime = metadata["current_datetime"] # '2022-09-30 00:00:00'
            current_date = current_datetime.split()[0] # '2022-09-30'
            year, month, day = map(int, current_date.split('-'))
            temporal = temporal.replace("CURRENT_DATE", f"{current_date}")
            temporal = temporal.replace("CURRENT_YEAR", f"{year}")
            temporal = temporal.replace("CURRENT_MONTH", f"{month}")
            temporal = temporal.replace("CURRENT_TIMESTAMP", f"{current_datetime}")
            # print(temporal, flush=True)
            temporal = [temporal]
        
        if spatials is not None:
            spatials = [f"'{name}'" for name in spatials]
        
        if "idu_name" in columns:
            columns.remove("idu_name")
            columns.append("raw:(SELECT name FROM idu_t WHERE id = data_t.idu_id) AS idu_name")

        if exp_tag == "woQM":
            return self.structured_query_to_query_string(
                table_name='data_t',
                columns=deepcopy(columns),
                conditions=deepcopy(temporal),
                subquery=f"idu_id IN (SELECT id FROM idu_t WHERE name IN ({', '.join(spatials)}))",
                get_rowids=get_rowids
            )
        else:
            return [self.structured_query_to_query_string(
                    table_name='data_t',
                    columns=deepcopy(columns),
                    conditions=deepcopy(temporal),
                    subquery=f"idu_id IN (SELECT id FROM idu_t WHERE name = {spatial})",
                    get_rowids=get_rowids
                ) for spatial in spatials]

    def structured_query_data_t(self, metadata, columns, temporal=None, spatials=None, get_rowids=False) -> pd.DataFrame:
        """
        temporal: '[~, ~]', '(~, ~)', '[~, ~)', '(~, ~]', '(~', '~)', '[~', '~]', '~'
        """

        query_strings = self.get_query_strings(metadata, columns, temporal, spatials, get_rowids)

        result = {}
        for spatial, query_string in zip(spatials, query_strings):
            r = self.execute_structured_query_string(query_string)
            if r is None:
                continue

            r["idu"] = spatial.replace("'", "")
            result[spatial] = r
        if len(result) == 0:
            return None
        df_result = pd.concat(result.values(), ignore_index=True)
        
        return df_result
    
    def create_continuous_aggregate(self, agg_name, select_query):
        """
        Creates a continuous aggregate in TimescaleDB.
        
        Parameters:
        - agg_name (str): The name of the aggregate view.
        - table_name (str): The name of the hypertable.
        - time_column (str): The name of the time column for the hypertable.
        - select_query (str): The SELECT query defining the aggregate.
        """
        try:
            self.cursor.execute(sql.SQL(
                "CREATE MATERIALIZED VIEW {} WITH (timescaledb.continuous) AS {}"
            ).format(
                sql.Identifier(agg_name),
                sql.SQL(select_query)
            ))
            logger.info(f"Continuous aggregate {agg_name} created successfully")
        except Exception as e:
            logger.error(f"An error occurred while creating continuous aggregate {agg_name}: {e}")

    def get_query_string(self, metadata, args):
        pass

    def set_retention_policy(self, table_name, retention_period):
        """
        Sets a data retention policy for a hypertable in TimescaleDB.
        
        Parameters:
        - table_name (str): The name of the hypertable.
        - retention_period (str): The retention period (e.g., '24 hours', '7 days').
        """
        try:
            self.cursor.execute(sql.SQL(
                "SELECT add_retention_policy({}, interval '{}')"
            ).format(sql.Identifier(table_name), sql.SQL(retention_period)))
            logger.info(f"Retention policy set for {table_name} with retention period {retention_period}")
        except Exception as e:
            logger.error(f"An error occurred while setting retention policy for {table_name}: {e}")

    def __del__(self):
        self.close()

    def close(self):
        if not self.cursor and self.is_connected:
            self.cursor.close()
            self.connection.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(parse_temporal('[~, ~]'))
    print(
        parse_temporal("[CURRENT_DATE - INTERVAL '1 day',  CURRENT_DATE)")
    )
    exit()
    
    # Exampe usage of the DBInstance class
    db_instance = DBInstance(dbname='PerSite_DB')
    
    if False:
        # Insert data into the hypertable
        data_to_insert = [
            {'idu_id': 1, 'roomtemp': 25.0, 'settemp': 22.0, 'oper': True, 'timestamp': '2022-01-01 00:00:00'},
            {'idu_id': 2, 'roomtemp': 26.0, 'settemp': 23.0, 'oper': False, 'timestamp': '2022-01-01 00:01:00'},
            {'idu_id': 1, 'roomtemp': 24.0, 'settemp': 21.0, 'oper': True, 'timestamp': '2022-01-01 00:02:00'},
            {'idu_id': 2, 'roomtemp': 27.0, 'settemp': 24.0, 'oper': False, 'timestamp': '2022-01-01 00:03:00'},
            {'idu_id': 1, 'roomtemp': 25.0, 'settemp': 22.0, 'oper': True, 'timestamp': '2022-01-01 00:04:00'},
            {'idu_id': 2, 'roomtemp': 26.0, 'settemp': 23.0, 'oper': False, 'timestamp': '2022-01-01 00:05:00'},
        ]
        db_instance.insert_data('data_t', data_to_insert)
    
    if True:
        import json
        
        print(
            db_instance.structured_query_to_query_string(
                'idu_t',
                columns=['id', 'name'],
                conditions=["name in ('01_IB5', '01_IB7')"]
            )
        )
        
        
        print(db_instance.structured_query_to_query_string(
            table_name='data_t',
            columns=['idu_id', 'roomtemp'],
            conditions=[
                "timestamp = '2022-09-30 00:00:00'",
                "roomtemp IS NOT NULL",
                "roomtemp IS DISTINCT FROM 'NaN'"
            ],
            subquery="idu_id in (SELECT id FROM idu_t WHERE name in ('01_IB5', '01_IB7'))"
        ))
        args = json.loads(
            """{
            "table_name": "data_t",
            "columns": ["idu_id", "timestamp", "roomtemp"],
            "conditions": [
                "timestamp BETWEEN '2022-09-30 00:00:00' AND '2022-09-30 00:59:59'",
                "roomtemp IS NOT NULL",
                "roomtemp IS DISTINCT FROM 'NaN'"
            ],
            "subquery": "idu_id IN (SELECT id FROM idu_t WHERE name IN ('01_IB5', '01_IB7'))"
            }"""
        )
        
        result = db_instance.structured_query_to_query_string(**args)
        # print(result[0]["timestamp"])
        print(result)
        
        # print(db_instance.structured_query(
        #     table_name='data_t',
        #     columns=['idu_id', 'roomtemp'],
        #     conditions=[
        #         "timestamp = '2022-09-30 00:00:00'",
        #         "roomtemp IS NOT NULL",
        #         "roomtemp IS DISTINCT FROM 'NaN'"
        #     ],
        #     subquery="idu_id = (SELECT id FROM idu_t WHERE name = '01_IB5')"
        # ))

        # get all idu_ids
        
    
    
    db_instance.close()
    
    