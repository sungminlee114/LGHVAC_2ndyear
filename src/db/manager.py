import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .instance import DBInstance

class DBManager:
    """
    This is a high-level interface which uses the DBInstance class to interact with the database.
    External modules should use this class to interact with the database.
    """
    def __init__(self):
        self.db_instance = DBInstance(dbname='PerSite_DB')
    
    def execute_sql(self, sql):
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
        return self.db_instance.execute_sql(sql)