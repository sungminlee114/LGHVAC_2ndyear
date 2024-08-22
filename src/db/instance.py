import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import psycopg2
from psycopg2 import sql


class DBInstance:
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


    def insert_data(self, table_name, data, ignore_if_exists=False):
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
        if self.cursor is not None and self.is_connected:
            self.cursor.close()
            self.connection.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    # Exampe usage of the DBInstance class
    db_instance = DBInstance(dbname='PerSite_DB')
    
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
    db_instance.close()