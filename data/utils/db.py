import duckdb


DB_PATH = '../data/db/master.db'


def get_db_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH)
