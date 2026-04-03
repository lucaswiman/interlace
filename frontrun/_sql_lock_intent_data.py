"""Lock intent classification data for SQL parsing.

Data tables mapping SQL lock clauses (FOR UPDATE, FOR SHARE, LOCK TABLE) and
advisory lock functions to :class:`LockIntent` values.  Keeping this separate
from the parsing logic makes it easy to see all lock-intent rules at a glance
and extend them without touching the parser.
"""

from __future__ import annotations

import enum
import re

# ---------------------------------------------------------------------------
# Lock intent enum
# ---------------------------------------------------------------------------


class LockIntent(enum.Enum):
    """Lock mode extracted from SQL statements (FOR UPDATE, FOR SHARE, LOCK TABLE)."""

    UPDATE = "UPDATE"
    SHARE = "SHARE"
    UPDATE_SKIP_LOCKED = "UPDATE_SKIP_LOCKED"


# ---------------------------------------------------------------------------
# Lock clause patterns
#
# Ordered from most-specific to least-specific so the first match wins.
# Used by both the regex fast-path and (indirectly) the sqlglot full parser.
# ---------------------------------------------------------------------------

_WS = r"[\s\n]+"

# fmt: off
_LOCK_CLAUSE_PATTERNS: list[tuple[re.Pattern[str], LockIntent]] = [
    (re.compile(r"\bFOR" + _WS + r"UPDATE" + _WS + r"SKIP" + _WS + r"LOCKED\b", re.I), LockIntent.UPDATE_SKIP_LOCKED),
    (re.compile(r"\bFOR" + _WS + r"UPDATE\b",                                    re.I), LockIntent.UPDATE),
    (re.compile(r"\bFOR" + _WS + r"SHARE\b",                                     re.I), LockIntent.SHARE),
]
# fmt: on

# ---------------------------------------------------------------------------
# Advisory lock functions
#
# Maps PostgreSQL/MySQL advisory-lock function names to their LockIntent.
# The get_lock() timeout argument is handled by the parser, not here.
# ---------------------------------------------------------------------------

_ADVISORY_LOCK_FUNCTIONS: dict[str, LockIntent] = {
    "pg_advisory_lock": LockIntent.UPDATE,
    "pg_advisory_xact_lock": LockIntent.UPDATE,
    "pg_advisory_lock_shared": LockIntent.SHARE,
    "pg_advisory_xact_lock_shared": LockIntent.SHARE,
    "get_lock": LockIntent.UPDATE,
}

# ---------------------------------------------------------------------------
# Lock intent merge priority
#
# When merging two lock intents (e.g. from a CTE and its outer query),
# the first entry in this list wins.
# ---------------------------------------------------------------------------

_LOCK_INTENT_PRIORITY: list[LockIntent] = [
    LockIntent.UPDATE,
    LockIntent.UPDATE_SKIP_LOCKED,
    LockIntent.SHARE,
]
