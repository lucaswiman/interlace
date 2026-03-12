import sqlite3

from frontrun.dpor import explore_dpor


def test_cross_column_conflict_fix():
    """Verify that DPOR finds conflicts between different column sets on the same table."""
    uri = "file:defect1_fix?mode=memory&cache=shared"
    # Keep one connection open to keep the database alive
    master = sqlite3.connect(uri, uri=True)
    master.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, active INT)")
    master.execute("DELETE FROM users")
    master.execute("INSERT INTO users VALUES (1, 'alice', 0)")
    master.commit()

    class State:
        def __init__(self):
            conn = sqlite3.connect(uri, uri=True)
            conn.execute("UPDATE users SET active = 0 WHERE id = 1")
            conn.commit()
            conn.close()
            self.activated = 0

    def thread_fn(s):
        conn = sqlite3.connect(uri, uri=True, isolation_level=None)
        c = conn.cursor()
        # Thread 1/2: SELECT by username (e.g. Django natural key lookup)
        c.execute("SELECT active FROM users WHERE username = 'alice'")
        row = c.fetchone()
        if row and not row[0]:
            # Thread 1/2: UPDATE by id (e.g. Django .save() after lookup)
            # This SHOULD conflict with the SELECT above in the other thread.
            c.execute("UPDATE users SET active = 1 WHERE id = 1")
            s.activated += 1
        conn.close()

    # We expect DPOR to find the interleaving where both SELECT before either UPDATE.
    # This leads to s.activated == 2, which fails the invariant.
    res = explore_dpor(
        setup=State,
        threads=[thread_fn, thread_fn],
        invariant=lambda s: s.activated < 2,
        detect_io=True,
    )

    # Bug fixed: res.property_holds should be False (race found)
    assert not res.property_holds, "DPOR missed the cross-column race condition!"
    master.close()


def test_row_level_preserved_for_primary_colset():
    """Verify that row-level benefits are still preserved for the primary column set."""
    uri = "file:row_level_preserved?mode=memory&cache=shared"
    master = sqlite3.connect(uri, uri=True)
    master.execute("CREATE TABLE IF NOT EXISTS counters (id INTEGER PRIMARY KEY, val INT)")
    master.execute("DELETE FROM counters")
    master.execute("INSERT INTO counters VALUES (1, 0), (2, 0)")
    master.commit()

    class State:
        def __init__(self):
            conn = sqlite3.connect(uri, uri=True)
            conn.execute("UPDATE counters SET val = 0")
            conn.commit()
            conn.close()

    def t1(s):
        conn = sqlite3.connect(uri, uri=True, isolation_level=None)
        # Update row 1
        conn.execute("UPDATE counters SET val = 1 WHERE id = 1")
        conn.close()

    def t2(s):
        conn = sqlite3.connect(uri, uri=True, isolation_level=None)
        # Update row 2
        conn.execute("UPDATE counters SET val = 1 WHERE id = 2")
        conn.close()

    # Since both use 'id' (the primary colset) and different rows, they should be independent.
    # DPOR should explore exactly 1 path.
    res = explore_dpor(
        setup=State,
        threads=[t1, t2],
        invariant=lambda s: True,
        detect_io=True,
    )

    assert res.num_explored == 1, f"Row-level benefits lost! Explored {res.num_explored} paths instead of 1."
    master.close()
