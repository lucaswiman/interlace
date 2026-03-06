"""Schema information for SQL conflict detection.

Provides structures to store and query database schema metadata, primarily
Foreign Key relationships, which are necessary for correct conflict detection
in multi-table workloads (Phase 6).

If a table T1 has a foreign key to T2, any write to T1 implicitly reads T2
(to validate the constraint).  Without this, operations on T1 and T2 might
appear independent when they are not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

@dataclass(frozen=True)
class ForeignKey:
    """A foreign key constraint."""
    table: str          # The child table (referencing)
    column: str         # The child column
    ref_table: str      # The parent table (referenced)
    ref_column: str     # The parent column


@dataclass
class Schema:
    """Registry of database schema information."""
    
    # Map from child_table -> list[ForeignKey]
    _fks: dict[str, list[ForeignKey]] = field(default_factory=dict)
    
    def add_foreign_key(self, fk: ForeignKey) -> None:
        """Register a foreign key constraint."""
        if fk.table not in self._fks:
            self._fks[fk.table] = []
        self._fks[fk.table].append(fk)
        
    def get_fks(self, table: str) -> list[ForeignKey]:
        """Get all foreign keys where the given table is the child."""
        return self._fks.get(table, [])

    def get_referenced_tables(self, table: str) -> set[str]:
        """Get set of tables referenced by this table via FKs."""
        return {fk.ref_table for fk in self.get_fks(table)}


# Global singleton for the application schema
_GLOBAL_SCHEMA = Schema()

def register_schema(schema: Schema) -> None:
    """Set the global schema instance."""
    global _GLOBAL_SCHEMA
    _GLOBAL_SCHEMA = schema

def get_schema() -> Schema:
    """Get the current global schema."""
    return _GLOBAL_SCHEMA
