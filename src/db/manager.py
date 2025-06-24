import logging
logger = logging.getLogger(__name__)

from .instance import DBInstance

class DBManager:
    """
    This is a high-level interface which uses the DBInstance class to interact with the database.
    External modules should use this class to interact with the database.
    """
    db_instance = DBInstance(
        dbname='PerSite_DB',
        host='1.233.219.93',
        port=9014,
    )
    
    @classmethod
    def execute_sql(cls, sql:str):
        """
        Executes an SQL query against the database.

        This method automatically determines whether to fetch results based on the query type.
        It fetches and returns results for SELECT queries, and commits changes for other queries
        like INSERT, UPDATE, DELETE, etc.

        Parameters:
        - sql (str): The SQL query to be executed.

        Returns:
        - list: A list of tuples containing the results if the query is a SELECT query.
        or
        - None: If the query is not a SELECT query, or if an error occurs.

        Logs:
        - Info: When the query is executed successfully.
        - Error: If an exception occurs during query execution.
        """
        return cls.db_instance.execute_sql(sql)

    @classmethod
    def structured_query_data_t(cls, metadata, args, get_rowids=False):
        return cls.db_instance.structured_query_data_t(metadata, **args, get_rowids=get_rowids)

    @classmethod
    def structured_query_data_t_v2(cls, metadata, m_raw, t_raw, s_raw, get_rowids=False):
        return cls.db_instance.structured_query_data_t_v2(metadata, m_raw, t_raw, s_raw, get_rowids=get_rowids)

    @classmethod
    def get_query_strings_v2(cls, metadata, m_raw, t_raw, s_raw, get_rowids=False):
        m_raw.append("idu_name")
        columns = list(set(m_raw))

        query_strings = cls.db_instance.get_query_strings_v2(metadata, columns, t_raw, s_raw, get_rowids)
        return query_strings

    @classmethod
    def get_query_strings(cls, metadata, args, exp_tag=None):
        return cls.db_instance.get_query_strings(metadata, **args, exp_tag=exp_tag)

    @classmethod
    def execute_structured_query_string(cls, query_string:str):
        return cls.db_instance.execute_structured_query_string(query_string)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)