# Algorithm 1.5: Parameter Resolution

Phase 2 row-level detection needs the *actual values* from WHERE predicates to determine whether two operations target the same row. But ORM-generated queries are nearly always parameterized:

```sql
-- What the ORM generates:
SELECT * FROM users WHERE id = %s          -- psycopg2 (format/pyformat)
SELECT * FROM users WHERE id = ?           -- sqlite3   (qmark)
SELECT * FROM users WHERE id = :id         -- cx_Oracle  (named)
SELECT * FROM users WHERE id = :1          -- oracledb   (numeric)
```

sqlglot parses `%s` as an identifier, `?` as a placeholder, `:id` as a session variable — none produce `exp.Literal` nodes. So `extract_equality_predicates` returns `[]` and we silently fall back to table-level. The query works, but we lose the row-level precision that Phase 2 is supposed to provide.

**Fix:** Before feeding SQL to `extract_equality_predicates`, substitute placeholders with the actual parameter values passed to `cursor.execute(operation, parameters)`. The resolved SQL is only used for AST analysis — it's never executed.

## 1.5a. Value Conversion

```python
def _python_to_sql_literal(value: Any) -> str:
    """Convert a Python value to a SQL literal for AST analysis.

    The result must be syntactically valid SQL that sqlglot can parse.
    It does NOT need to be safe for execution.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"X'{bytes(value).hex()}'"
    # Default: string literal (double internal quotes)
    s = str(value).replace("'", "''")
    return f"'{s}'"
```

## 1.5b. Paramstyle Detection

PEP 249 requires every DBAPI module to expose a `paramstyle` attribute. We read it at patch time and bake it into the `_patched_execute` closure:

| `paramstyle` | Placeholder | Parameters | Drivers |
|---|---|---|---|
| `qmark` | `?` | sequence | sqlite3 |
| `numeric` | `:1`, `:2` | sequence | oracledb |
| `named` | `:name` | mapping | oracledb |
| `format` | `%s` | sequence | — |
| `pyformat` | `%(name)s` | mapping | psycopg2, pymysql |

**Subtlety:** psycopg2 and pymysql declare `paramstyle = "pyformat"` but in practice nearly all queries use `%s` with tuple parameters (plain `format` style). The resolution function detects this from the parameter type: `dict` → try `%(name)s` patterns, sequence → try `%s` patterns. This covers both styles with no driver-specific hacks.

## 1.5c. Resolution Functions

```python
import re

# Precompiled patterns.  Negative lookbehinds prevent matching escaped
# sequences (:: in PG casts, %% in psycopg2 literal-percent escapes).
_RE_QMARK    = re.compile(r"\?")
_RE_NUMERIC  = re.compile(r":(\d+)")
_RE_NAMED    = re.compile(r"(?<!:):([A-Za-z_]\w*)")
_RE_FORMAT   = re.compile(r"(?<!%)%s")
_RE_PYFORMAT = re.compile(r"(?<!%)%\(([^)]+)\)s")


def resolve_parameters(sql: str, parameters: Any, paramstyle: str) -> str:
    """Substitute parameter placeholders with SQL literal values.

    Supports all five PEP 249 paramstyles.  Returns the original SQL
    unchanged if resolution fails (wrong paramstyle, missing params, etc.).
    """
    if parameters is None:
        return sql
    try:
        if paramstyle == "qmark":
            return _resolve_positional(sql, parameters, _RE_QMARK)
        if paramstyle == "numeric":
            return _resolve_numeric(sql, parameters)
        if paramstyle == "named":
            return _resolve_named(sql, parameters)
        if paramstyle in ("format", "pyformat"):
            # Detect actual style from parameter type:
            # dict → %(name)s, sequence → %s
            if isinstance(parameters, dict):
                return _resolve_pyformat(sql, parameters)
            return _resolve_positional(sql, parameters, _RE_FORMAT)
    except (IndexError, KeyError, TypeError, ValueError):
        pass
    return sql


def _resolve_positional(sql: str, parameters: Any,
                         pattern: re.Pattern[str]) -> str:
    """Replace positional placeholders (? or %s) in left-to-right order."""
    params = tuple(parameters)
    idx = 0

    def replacer(m: re.Match[str]) -> str:
        nonlocal idx
        if idx >= len(params):
            return m.group(0)  # more placeholders than params → leave as-is
        val = _python_to_sql_literal(params[idx])
        idx += 1
        return val

    return pattern.sub(replacer, sql)


def _resolve_numeric(sql: str, parameters: Any) -> str:
    """Replace :N placeholders (1-based index)."""
    params = tuple(parameters)

    def replacer(m: re.Match[str]) -> str:
        idx = int(m.group(1)) - 1
        return _python_to_sql_literal(params[idx])

    return _RE_NUMERIC.sub(replacer, sql)


def _resolve_named(sql: str, parameters: Any) -> str:
    """Replace :name placeholders with named parameters."""
    params = dict(parameters)

    def replacer(m: re.Match[str]) -> str:
        return _python_to_sql_literal(params[m.group(1)])

    return _RE_NAMED.sub(replacer, sql)


def _resolve_pyformat(sql: str, parameters: Any) -> str:
    """Replace %(name)s placeholders with named parameters."""
    params = dict(parameters)

    def replacer(m: re.Match[str]) -> str:
        return _python_to_sql_literal(params[m.group(1)])

    return _RE_PYFORMAT.sub(replacer, sql)
```

## 1.5d. End-to-End Example

```python
# psycopg2 (paramstyle="pyformat", but uses %s with tuple params):
sql = "SELECT * FROM users WHERE id = %s AND region = %s"
params = (42, "us-east")

resolved = resolve_parameters(sql, params, "pyformat")
# → "SELECT * FROM users WHERE id = 42 AND region = 'us-east'"

preds = extract_equality_predicates(resolved)
# → [EqualityPredicate("id", "42"), EqualityPredicate("region", "us-east")]

resource_id = _sql_resource_id("users", preds)
# → "sql:users:(('id', '42'), ('region', 'us-east'))"
```

## 1.5e. Limitations

1. **Placeholders inside string literals.** `SELECT * FROM t WHERE name = 'What?' AND id = ?` — the regex matches the `?` inside the string literal too. In practice this doesn't happen: if the SQL has string literals, those values would be parameterized. If it does happen, sqlglot fails to parse the resolved SQL and we fall back to table-level (safe).

2. **`executemany` skips resolution.** `executemany` passes a *sequence* of parameter sets, each potentially targeting a different row. We skip parameter resolution entirely and use table-level (conservative, correct).

3. **Non-standard parameter styles.** Some drivers support styles beyond their declared `paramstyle` (e.g. psycopg3 uses `$1` internally for server-side prepared statements). We only handle the five PEP 249 styles. Unrecognized placeholders pass through unchanged → table-level fallback.

**Complexity:** O(n) regex substitution on the SQL string, plus O(k) for converting k parameter values to literals. Negligible vs. network RTT.
