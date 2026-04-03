"""Redis command classification for key-level conflict detection.

Classifies Redis commands into read/write operations and extracts the
key(s) each command accesses.  This enables DPOR to detect key-level
conflicts between threads/tasks operating on the same Redis keys.

Key extraction rules are derived from the official Redis ``commands.json``
key specifications (``begin_search`` / ``find_keys`` / access flags).  A
generic interpreter replaces per-command special-case logic.  Only a
handful of overrides remain for DPOR-specific semantics (transaction
control, pub/sub channel prefixing, Lua script atomicity).
"""

from __future__ import annotations

from typing import NamedTuple

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


class RedisAccessResult(NamedTuple):
    """Result of classifying a Redis command."""

    read_keys: list[str]
    write_keys: list[str]
    is_transaction_control: bool  # MULTI, EXEC, DISCARD, WATCH, UNWATCH


# ---------------------------------------------------------------------------
# Key-spec data table — derived from Redis commands.json
#
# Each command maps to a list of key-spec entries:
#   (begin_search, find_keys, is_read, is_write)
#
# begin_search formats:
#   ("idx", N)                  — keys start at argument index N (1-based
#                                 from the full Redis protocol; we subtract
#                                 1 to index into cmd_args)
#   ("kw", keyword, startfrom)  — keys follow the first occurrence of
#                                 *keyword* (searched from ``startfrom``)
#
# find_keys formats:
#   ("rng", lastkey, keystep, limit)
#       lastkey=0  → single key
#       lastkey=-1 → all remaining args (limit divides count if >1)
#       lastkey=-2 → all args except the last
#       lastkey=N>0 → N additional keys after the first
#       keystep    → stride between keys (1=every arg, 2=every other)
#       limit      → divisor for remaining-arg count (0 or 1 = no limit)
#   ("kn", keynumidx, firstkey, keystep)
#       keynumidx → offset from begin to the arg holding the key count
#       firstkey  → offset from begin to the first actual key
#       keystep   → stride between keys
# ---------------------------------------------------------------------------

# fmt: off
_KeySpec = tuple[tuple[str | int, ...], tuple[str | int, ...], bool, bool]
_COMMAND_KEY_SPECS: dict[str, list[_KeySpec]] = {
    "APPEND": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "BITCOUNT": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "BITFIELD": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "BITFIELD_RO": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "BITOP": [(("idx", 2), ("rng", 0, 1, 0), False, True), (("idx", 3), ("rng", -1, 1, 0), True, False)],
    "BITPOS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "BLMOVE": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), True, True)],
    "BLMPOP": [(("idx", 2), ("kn", 0, 1, 1), True, True)],
    "BLPOP": [(("idx", 1), ("rng", -2, 1, 0), True, True)],
    "BRPOP": [(("idx", 1), ("rng", -2, 1, 0), True, True)],
    "BRPOPLPUSH": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), True, True)],
    "BZMPOP": [(("idx", 2), ("kn", 0, 1, 1), True, True)],
    "BZPOPMAX": [(("idx", 1), ("rng", -2, 1, 0), True, True)],
    "BZPOPMIN": [(("idx", 1), ("rng", -2, 1, 0), True, True)],
    "COPY": [(("idx", 1), ("rng", 0, 1, 0), True, False), (("idx", 2), ("rng", 0, 1, 0), False, True)],
    "DECR": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "DECRBY": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "DEL": [(("idx", 1), ("rng", -1, 1, 0), False, True)],
    "DUMP": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "EXISTS": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "EXPIRE": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "EXPIREAT": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "EXPIRETIME": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "FCALL": [(("idx", 2), ("kn", 0, 1, 1), True, True)],
    "FCALL_RO": [(("idx", 2), ("kn", 0, 1, 1), True, False)],
    "GEOADD": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "GEODIST": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GEOHASH": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GEOPOS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GEORADIUS": [(("idx", 1), ("rng", 0, 1, 0), True, False), (("kw", "STORE", 6), ("rng", 0, 1, 0), False, True), (("kw", "STOREDIST", 6), ("rng", 0, 1, 0), False, True)],
    "GEORADIUSBYMEMBER": [(("idx", 1), ("rng", 0, 1, 0), True, False), (("kw", "STORE", 5), ("rng", 0, 1, 0), False, True), (("kw", "STOREDIST", 5), ("rng", 0, 1, 0), False, True)],
    "GEORADIUSBYMEMBER_RO": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GEORADIUS_RO": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GEOSEARCH": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GEOSEARCHSTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("rng", 0, 1, 0), True, False)],
    "GET": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GETBIT": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GETDEL": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "GETEX": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "GETRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GETSET": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HDEL": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HEXISTS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HGET": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HGETALL": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HINCRBY": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HINCRBYFLOAT": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HKEYS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HLEN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HMGET": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HMSET": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HRANDFIELD": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HSCAN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HSET": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HSETNX": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HSTRLEN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HVALS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "INCR": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "INCRBY": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "INCRBYFLOAT": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LCS": [(("idx", 1), ("rng", 1, 1, 0), True, False)],
    "LINDEX": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "LINSERT": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LLEN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "LMOVE": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), True, True)],
    "LMPOP": [(("idx", 1), ("kn", 0, 1, 1), True, True)],
    "LPOP": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LPOS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "LPUSH": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LPUSHX": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "LREM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LSET": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LTRIM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "MEMORY USAGE": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "MGET": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "MIGRATE": [(("idx", 3), ("rng", 0, 1, 0), True, True), (("kw", "KEYS", -2), ("rng", -1, 1, 0), True, True)],
    "MOVE": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "MSET": [(("idx", 1), ("rng", -1, 2, 0), False, True)],
    "MSETNX": [(("idx", 1), ("rng", -1, 2, 0), False, True)],
    "OBJECT ENCODING": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "OBJECT FREQ": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "OBJECT IDLETIME": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "OBJECT REFCOUNT": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "PERSIST": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "PEXPIRE": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "PEXPIREAT": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "PEXPIRETIME": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "PFADD": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "PFCOUNT": [(("idx", 1), ("rng", -1, 1, 0), True, True)],
    "PFDEBUG": [(("idx", 2), ("rng", 0, 1, 0), True, True)],
    "PFMERGE": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", -1, 1, 0), True, False)],
    "PSETEX": [(("idx", 1), ("rng", 0, 1, 0), False, True)],
    "PTTL": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "RENAME": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), False, True)],
    "RENAMENX": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), False, True)],
    "RESTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True)],
    "RESTORE-ASKING": [(("idx", 1), ("rng", 0, 1, 0), False, True)],
    "RPOP": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "RPOPLPUSH": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), True, True)],
    "RPUSH": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "RPUSHX": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SADD": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SCARD": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SDIFF": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "SDIFFSTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("rng", -1, 1, 0), True, False)],
    "SET": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SETBIT": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SETEX": [(("idx", 1), ("rng", 0, 1, 0), False, True)],
    "SETNX": [(("idx", 1), ("rng", 0, 1, 0), False, True)],
    "SETRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SINTER": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "SINTERCARD": [(("idx", 1), ("kn", 0, 1, 1), True, False)],
    "SINTERSTORE": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", -1, 1, 0), True, False)],
    "SISMEMBER": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SMEMBERS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SMISMEMBER": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SMOVE": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), True, True)],
    "SORT": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SORT_RO": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SPOP": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SRANDMEMBER": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SREM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SSCAN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "STRLEN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SUBSTR": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SUNION": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "SUNIONSTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("rng", -1, 1, 0), True, False)],
    "TOUCH": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "TTL": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "TYPE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "UNLINK": [(("idx", 1), ("rng", -1, 1, 0), False, True)],
    "WATCH": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "XACK": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "XADD": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "XAUTOCLAIM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "XCLAIM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "XDEL": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "XGROUP CREATE": [(("idx", 2), ("rng", 0, 1, 0), True, True)],
    "XGROUP CREATECONSUMER": [(("idx", 2), ("rng", 0, 1, 0), True, True)],
    "XGROUP DELCONSUMER": [(("idx", 2), ("rng", 0, 1, 0), True, True)],
    "XGROUP DESTROY": [(("idx", 2), ("rng", 0, 1, 0), True, True)],
    "XGROUP SETID": [(("idx", 2), ("rng", 0, 1, 0), True, True)],
    "XINFO CONSUMERS": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "XINFO GROUPS": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "XINFO STREAM": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "XLEN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "XPENDING": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "XRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "XREAD": [(("kw", "STREAMS", 1), ("rng", -1, 1, 2), True, False)],
    "XREADGROUP": [(("kw", "STREAMS", 4), ("rng", -1, 1, 2), True, True)],
    "XREVRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "XSETID": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "XTRIM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZADD": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZCARD": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZCOUNT": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZDIFF": [(("idx", 1), ("kn", 0, 1, 1), True, False)],
    "ZDIFFSTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("kn", 0, 1, 1), True, False)],
    "ZINCRBY": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZINTER": [(("idx", 1), ("kn", 0, 1, 1), True, False)],
    "ZINTERCARD": [(("idx", 1), ("kn", 0, 1, 1), True, False)],
    "ZINTERSTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("kn", 0, 1, 1), True, False)],
    "ZLEXCOUNT": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZMPOP": [(("idx", 1), ("kn", 0, 1, 1), True, True)],
    "ZMSCORE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZPOPMAX": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZPOPMIN": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZRANDMEMBER": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZRANGEBYLEX": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZRANGEBYSCORE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZRANGESTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("rng", 0, 1, 0), True, False)],
    "ZRANK": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZREM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZREMRANGEBYLEX": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZREMRANGEBYRANK": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZREMRANGEBYSCORE": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZREVRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZREVRANGEBYLEX": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZREVRANGEBYSCORE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZREVRANK": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZSCAN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZSCORE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZUNION": [(("idx", 1), ("kn", 0, 1, 1), True, False)],
    "ZUNIONSTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("kn", 0, 1, 1), True, False)],
}
# fmt: on

# Transaction control commands — no keys, flag only.
_TX_CONTROL_CMDS: frozenset[str] = frozenset({"MULTI", "EXEC", "DISCARD", "UNWATCH"})

# EVAL/EVALSHA are atomic (Lua scripts execute without interleaving).
# Treat as transaction control with no key-level accesses.  See defect #8.
_EVAL_CMDS: frozenset[str] = frozenset({"EVAL", "EVALSHA", "EVAL_RO", "EVALSHA_RO", "FCALL", "FCALL_RO"})

# Server/connection/cluster commands that never operate on specific keys.
# These must not hit the conservative fallback (which treats arg[0] as a key).
_NO_KEY_CMDS: frozenset[str] = frozenset(
    {
        "ACL",
        "ASKING",
        "AUTH",
        "BGREWRITEAOF",
        "BGSAVE",
        "CLIENT",
        "CLUSTER",
        "COMMAND",
        "CONFIG",
        "DBSIZE",
        "DEBUG",
        "ECHO",
        "FAILOVER",
        "FLUSHALL",
        "FLUSHDB",
        "FUNCTION",
        "HELLO",
        "INFO",
        "KEYS",
        "LASTSAVE",
        "LATENCY",
        "LOLWUT",
        "MODULE",
        "MONITOR",
        "PFSELFTEST",
        "PING",
        "PSYNC",
        "PUBSUB",
        "QUIT",
        "RANDOMKEY",
        "READONLY",
        "READWRITE",
        "REPLCONF",
        "REPLICAOF",
        "RESET",
        "ROLE",
        "SAVE",
        "SCAN",
        "SCRIPT",
        "SELECT",
        "SHUTDOWN",
        "SLAVEOF",
        "SLOWLOG",
        "SWAPDB",
        "SYNC",
        "TIME",
        "WAIT",
        "WAITAOF",
    }
)

# Parent commands that dispatch to subcommands in the key-spec table.
# When a specific subcommand (e.g. "OBJECT HELP") is not found, we return
# no-keys instead of hitting the conservative fallback.
_SUBCOMMAND_PARENTS: frozenset[str] = frozenset(
    {parent for key in _COMMAND_KEY_SPECS if " " in key for parent in [key.split(" ", 1)[0]]}
)


# ---------------------------------------------------------------------------
# Generic key-spec interpreter
# ---------------------------------------------------------------------------


def _extract_keys_from_spec(
    cmd_args: tuple[object, ...],
    begin: tuple[str | int, ...],
    find: tuple[str | int, ...],
) -> list[str]:
    """Extract keys from *cmd_args* using a single key-spec entry.

    The ``begin`` tuple locates the starting position; ``find`` enumerates
    keys from that position.  Both formats are documented above.

    Note: ``cmd_args`` does **not** include the command name.  The indices
    in the key-spec table are 1-based (position 1 = first argument after
    the command name), so we subtract 1 when indexing into *cmd_args*.
    """
    n_args = len(cmd_args)

    # --- Resolve begin position (0-based into cmd_args) ---
    if begin[0] == "idx":
        start = int(begin[1]) - 1  # 1-based → 0-based
        if start < 0 or start >= n_args:
            return []
    elif begin[0] == "kw":
        keyword = str(begin[1]).upper()
        startfrom = int(begin[2])
        # startfrom can be negative (search from end); convert to 0-based cmd_args index.
        if startfrom >= 0:
            search_start = max(startfrom - 1, 0)
        else:
            search_start = max(n_args + startfrom, 0)
        args_upper = [str(a).upper() for a in cmd_args]
        found = -1
        for j in range(search_start, n_args):
            if args_upper[j] == keyword:
                found = j
                break
        if found < 0:
            return []
        start = found + 1  # keys begin after the keyword
        if start >= n_args:
            return []
    else:
        return []

    # --- Enumerate keys from *start* ---
    if find[0] == "rng":
        lastkey = int(find[1])
        keystep = max(int(find[2]), 1)
        limit = int(find[3])

        if lastkey == 0:
            # Single key at start.
            return [str(cmd_args[start])]
        elif lastkey > 0:
            # Fixed number of additional keys.
            end = min(start + lastkey, n_args - 1)
            return [str(cmd_args[i]) for i in range(start, end + 1, keystep)]
        else:
            # lastkey < 0: keys extend to the end (adjusted by lastkey).
            # lastkey=-1 → up to and including the last arg
            # lastkey=-2 → up to and including the second-to-last arg
            end = n_args + lastkey + 1  # exclusive upper bound
            if end <= start:
                return []
            count = end - start
            # Apply limit divisor (e.g. XREAD: limit=2 means first half are keys).
            if limit >= 2:
                count = count // limit
            if count <= 0:
                return []
            return [str(cmd_args[start + i * keystep]) for i in range(count) if start + i * keystep < n_args]

    elif find[0] == "kn":
        keynumidx = int(find[1])
        firstkey_offset = int(find[2])
        keystep = max(int(find[3]), 1)

        numkeys_pos = start + keynumidx
        if numkeys_pos >= n_args:
            return []
        try:
            numkeys = int(str(cmd_args[numkeys_pos]))
        except (ValueError, TypeError):
            return []

        first_pos = start + firstkey_offset
        keys: list[str] = []
        for i in range(numkeys):
            pos = first_pos + i * keystep
            if pos >= n_args:
                break
            keys.append(str(cmd_args[pos]))
        return keys

    return []


def _lookup_key_specs(cmd_name_upper: str, cmd_args: tuple[object, ...]) -> list[_KeySpec] | None:
    """Look up key specs, trying subcommand dispatch first."""
    if cmd_args:
        sub = cmd_name_upper + " " + str(cmd_args[0]).upper()
        specs = _COMMAND_KEY_SPECS.get(sub)
        if specs is not None:
            return specs
    return _COMMAND_KEY_SPECS.get(cmd_name_upper)


# ---------------------------------------------------------------------------
# SORT special handling (STORE keyword has "unknown" key_spec in Redis)
# ---------------------------------------------------------------------------


def _parse_sort(cmd_args: tuple[object, ...]) -> RedisAccessResult:
    """Handle SORT which has an optional STORE dest that can't be expressed
    via static key_specs (Redis marks it as ``unknown``)."""
    if not cmd_args:
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)

    first_key = str(cmd_args[0])
    # Walk args, skipping values after BY, GET, LIMIT to avoid treating
    # user-supplied patterns (e.g. "store") as the STORE keyword.
    i = 1
    while i < len(cmd_args):
        token = str(cmd_args[i]).upper()
        if token in ("BY", "GET"):
            i += 2
        elif token == "LIMIT":
            i += 3
        elif token == "STORE" and i + 1 < len(cmd_args):
            dest = str(cmd_args[i + 1])
            return RedisAccessResult(read_keys=[first_key], write_keys=[dest], is_transaction_control=False)
        else:
            i += 1
    return RedisAccessResult(read_keys=[first_key], write_keys=[], is_transaction_control=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_redis_access(cmd_name: str, cmd_args: tuple[object, ...]) -> RedisAccessResult:
    """Classify a Redis command and extract the key(s) it accesses.

    Args:
        cmd_name: The Redis command name (e.g. "GET", "SET", "HSET").
        cmd_args: The command arguments (after the command name).

    Returns:
        A ``RedisAccessResult`` with the read/write key sets.
    """
    upper = cmd_name.upper()

    # --- Transaction control (no keys) ---
    if upper in _TX_CONTROL_CMDS:
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=True)

    # WATCH — reads keys for conflict detection, flagged as transaction control.
    if upper == "WATCH":
        keys = [str(a) for a in cmd_args]
        return RedisAccessResult(read_keys=keys, write_keys=[], is_transaction_control=True)

    # EVAL/EVALSHA/FCALL — atomic scripts, treated as transaction control (defect #8).
    if upper in _EVAL_CMDS:
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=True)

    # Pub/Sub — channels, not keys.  Prefix with "channel:" to distinguish.
    # Includes sharded pub/sub (SPUBLISH, SSUBSCRIBE, SUNSUBSCRIBE).
    if upper in ("PUBLISH", "SPUBLISH") and cmd_args:
        return RedisAccessResult(read_keys=[], write_keys=[f"channel:{cmd_args[0]}"], is_transaction_control=False)
    if upper in ("SUBSCRIBE", "PSUBSCRIBE", "SSUBSCRIBE"):
        return RedisAccessResult(
            read_keys=[f"channel:{a}" for a in cmd_args], write_keys=[], is_transaction_control=False
        )
    if upper in ("UNSUBSCRIBE", "PUNSUBSCRIBE", "SUNSUBSCRIBE"):
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)

    # SORT/SORT_RO — STORE keyword has "unknown" key_spec; use dedicated parser.
    if upper in ("SORT", "SORT_RO"):
        return _parse_sort(cmd_args)

    # Server/connection commands — no key-level conflicts.
    if upper in _NO_KEY_CMDS:
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)

    # --- Generic key-spec extraction ---
    specs = _lookup_key_specs(upper, cmd_args)

    if specs is None:
        # If this is a parent that has subcommand variants (e.g. OBJECT, XGROUP)
        # but the specific subcommand wasn't found, return no keys.
        if upper in _SUBCOMMAND_PARENTS:
            return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)
        # Truly unknown command with args → conservative fallback: treat first arg as write key.
        if cmd_args:
            return RedisAccessResult(read_keys=[], write_keys=[str(cmd_args[0])], is_transaction_control=False)
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)

    read_keys: list[str] = []
    write_keys: list[str] = []
    for begin, find, is_read, is_write in specs:
        keys = _extract_keys_from_spec(cmd_args, begin, find)
        if is_read:
            read_keys.extend(keys)
        if is_write:
            write_keys.extend(keys)

    return RedisAccessResult(read_keys=read_keys, write_keys=write_keys, is_transaction_control=False)
