import sqlite3
import time

def execute_with_retry(conn, sql, params=None):
    for i in range(50):
        try:
            if params:
                return conn.execute(sql, params)
            else:
                return conn.execute(sql)
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(0.01 * (i + 1))
                continue
            raise
    if params:
        return conn.execute(sql, params)
    else:
        return conn.execute(sql)
