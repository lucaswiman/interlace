from frontrun._sql_parsing import parse_sql_access

def test_repro_bug_2_delete_with_placeholders():
    # DELETE: fails — returns empty tables
    sql = 'DELETE FROM "t" WHERE "t"."id" IN (%s)'
    result = parse_sql_access(sql)
    print(f"DELETE result: {result}")
    assert 't' in result.read_tables
    assert 't' in result.write_tables

def test_repro_bug_2_insert_with_placeholders():
    # INSERT: fails — returns empty tables
    sql = 'INSERT INTO "t" ("a", "b") VALUES (%s, %s) RETURNING "t"."id"'
    result = parse_sql_access(sql)
    print(f"INSERT result: {result}")
    assert result.read_tables == set()
    assert 't' in result.write_tables
