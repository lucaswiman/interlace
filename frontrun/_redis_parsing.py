"""Redis command classification for key-level conflict detection.

Classifies Redis commands into read/write operations and extracts the
key(s) each command accesses.  This enables DPOR to detect key-level
conflicts between threads/tasks operating on the same Redis keys.

The classification covers all major Redis command groups:
- String commands (GET, SET, INCR, APPEND, etc.)
- Hash commands (HGET, HSET, HDEL, etc.)
- List commands (LPUSH, RPUSH, LPOP, RPOP, LRANGE, etc.)
- Set commands (SADD, SREM, SMEMBERS, etc.)
- Sorted set commands (ZADD, ZREM, ZRANGE, etc.)
- Key commands (DEL, EXISTS, EXPIRE, RENAME, TYPE, etc.)
- HyperLogLog commands (PFADD, PFCOUNT, PFMERGE)
- Pub/Sub commands (PUBLISH, SUBSCRIBE — treated as channel resources)
- Stream commands (XADD, XREAD, etc.)
- Script/transaction commands (EVAL, MULTI/EXEC)
- Cluster and server commands (ignored — no key-level conflicts)
"""

from __future__ import annotations

from typing import NamedTuple


class RedisAccessResult(NamedTuple):
    """Result of classifying a Redis command."""

    read_keys: list[str]
    write_keys: list[str]
    is_transaction_control: bool  # MULTI, EXEC, DISCARD, WATCH, UNWATCH


# ---------------------------------------------------------------------------
# Command classification tables
# ---------------------------------------------------------------------------

# Commands that read a single key (first argument).
_SINGLE_KEY_READ_CMDS: frozenset[str] = frozenset(
    {
        "GET",
        "STRLEN",
        "GETRANGE",
        "SUBSTR",
        "HGET",
        "HGETALL",
        "HKEYS",
        "HVALS",
        "HLEN",
        "HEXISTS",
        "HMGET",
        "HRANDFIELD",
        "HSCAN",
        "LLEN",
        "LRANGE",
        "LINDEX",
        "LPOS",
        "SCARD",
        "SISMEMBER",
        "SMISMEMBER",
        "SMEMBERS",
        "SRANDMEMBER",
        "SSCAN",
        "ZCARD",
        "ZCOUNT",
        "ZLEXCOUNT",
        "ZRANGE",
        "ZRANGEBYLEX",
        "ZRANGEBYSCORE",
        "ZRANK",
        "ZREVRANGE",
        "ZREVRANGEBYLEX",
        "ZREVRANGEBYSCORE",
        "ZREVRANK",
        "ZSCORE",
        "ZMSCORE",
        "ZRANDMEMBER",
        "ZSCAN",
        "TYPE",
        "TTL",
        "PTTL",
        "PERSIST",
        "DUMP",
        "OBJECT",
        "DEBUG",
        "XLEN",
        "XRANGE",
        "XREVRANGE",
        "XINFO",
        "XPENDING",
        "PFCOUNT",
        "GETEX",
        "EXPIRETIME",
        "PEXPIRETIME",
    }
)

# Commands that write a single key (first argument).
_SINGLE_KEY_WRITE_CMDS: frozenset[str] = frozenset(
    {
        "SET",
        "SETNX",
        "SETEX",
        "PSETEX",
        "SETRANGE",
        "APPEND",
        "INCR",
        "INCRBY",
        "INCRBYFLOAT",
        "DECR",
        "DECRBY",
        "HSET",
        "HSETNX",
        "HMSET",
        "HINCRBY",
        "HINCRBYFLOAT",
        "HDEL",
        "LPUSH",
        "LPUSHX",
        "RPUSH",
        "RPUSHX",
        "LPOP",
        "RPOP",
        "LSET",
        "LINSERT",
        "LREM",
        "LTRIM",
        "SADD",
        "SREM",
        "SPOP",
        "ZADD",
        "ZREM",
        "ZINCRBY",
        "ZREMRANGEBYLEX",
        "ZREMRANGEBYRANK",
        "ZREMRANGEBYSCORE",
        "ZPOPMIN",
        "ZPOPMAX",
        "EXPIRE",
        "EXPIREAT",
        "PEXPIRE",
        "PEXPIREAT",
        "XADD",
        "XDEL",
        "XTRIM",
        "XACK",
        "PFADD",
        "LMPOP",
    }
)

# Commands that read+write a single key (first argument).
# These are commands that both read and modify state atomically.
_SINGLE_KEY_READ_WRITE_CMDS: frozenset[str] = frozenset(
    {
        "GETSET",
        "GETDEL",
    }
)

# Commands that take multiple keys as arguments (all positional args are keys).
_MULTI_KEY_READ_CMDS: frozenset[str] = frozenset(
    {
        "MGET",
        "EXISTS",
    }
)

_MULTI_KEY_WRITE_CMDS: frozenset[str] = frozenset(
    {
        "DEL",
        "UNLINK",
    }
)

# Transaction control commands (no keys).
_TX_CONTROL_CMDS: frozenset[str] = frozenset(
    {
        "MULTI",
        "EXEC",
        "DISCARD",
    }
)


def parse_redis_access(cmd_name: str, cmd_args: tuple[object, ...]) -> RedisAccessResult:
    """Classify a Redis command and extract the key(s) it accesses.

    Args:
        cmd_name: The Redis command name (e.g. "GET", "SET", "HSET").
        cmd_args: The command arguments (after the command name).

    Returns:
        A ``RedisAccessResult`` with the read/write key sets.
    """
    upper = cmd_name.upper()

    # Transaction control — no keys.
    if upper in _TX_CONTROL_CMDS:
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=True)

    # WATCH/UNWATCH — reads for conflict purposes.
    if upper == "WATCH":
        keys = [str(a) for a in cmd_args]
        return RedisAccessResult(read_keys=keys, write_keys=[], is_transaction_control=True)
    if upper == "UNWATCH":
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=True)

    # No arguments → server/connection command (PING, INFO, etc.)
    if not cmd_args:
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)

    first_key = str(cmd_args[0])

    # Single-key read commands.
    if upper in _SINGLE_KEY_READ_CMDS:
        return RedisAccessResult(read_keys=[first_key], write_keys=[], is_transaction_control=False)

    # Single-key write commands.
    if upper in _SINGLE_KEY_WRITE_CMDS:
        return RedisAccessResult(read_keys=[], write_keys=[first_key], is_transaction_control=False)

    # Single-key read+write commands.
    if upper in _SINGLE_KEY_READ_WRITE_CMDS:
        return RedisAccessResult(read_keys=[first_key], write_keys=[first_key], is_transaction_control=False)

    # Multi-key read commands.
    if upper in _MULTI_KEY_READ_CMDS:
        keys = [str(a) for a in cmd_args]
        return RedisAccessResult(read_keys=keys, write_keys=[], is_transaction_control=False)

    # Multi-key write commands.
    if upper in _MULTI_KEY_WRITE_CMDS:
        keys = [str(a) for a in cmd_args]
        return RedisAccessResult(read_keys=[], write_keys=keys, is_transaction_control=False)

    # MSET / MSETNX — args are key-value pairs.
    if upper in ("MSET", "MSETNX"):
        keys = [str(cmd_args[i]) for i in range(0, len(cmd_args), 2)]
        return RedisAccessResult(read_keys=[], write_keys=keys, is_transaction_control=False)

    # RENAME / RENAMENX — reads+deletes source, writes destination.
    if upper in ("RENAME", "RENAMENX") and len(cmd_args) >= 2:
        return RedisAccessResult(
            read_keys=[first_key],
            write_keys=[first_key, str(cmd_args[1])],
            is_transaction_control=False,
        )

    # RPOPLPUSH / LMOVE — pops from source (read+write), pushes to destination (write).
    if upper in ("RPOPLPUSH", "LMOVE", "BRPOPLPUSH", "BLMOVE") and len(cmd_args) >= 2:
        return RedisAccessResult(
            read_keys=[first_key],
            write_keys=[first_key, str(cmd_args[1])],
            is_transaction_control=False,
        )

    # SMOVE — removes from source (read+write), adds to destination (write).
    if upper == "SMOVE" and len(cmd_args) >= 2:
        return RedisAccessResult(
            read_keys=[first_key],
            write_keys=[first_key, str(cmd_args[1])],
            is_transaction_control=False,
        )

    # Blocking pop commands — first arg(s) are keys, last is timeout.
    if upper in ("BLPOP", "BRPOP"):
        # Args: key [key ...] timeout
        keys = [str(a) for a in cmd_args[:-1]] if len(cmd_args) > 1 else [first_key]
        return RedisAccessResult(read_keys=[], write_keys=keys, is_transaction_control=False)

    # BZPOPMIN / BZPOPMAX — key [key ...] timeout
    if upper in ("BZPOPMIN", "BZPOPMAX"):
        keys = [str(a) for a in cmd_args[:-1]] if len(cmd_args) > 1 else [first_key]
        return RedisAccessResult(read_keys=[], write_keys=keys, is_transaction_control=False)

    # Set operations with destination: SUNIONSTORE, SINTERSTORE, SDIFFSTORE
    if upper in ("SUNIONSTORE", "SINTERSTORE", "SDIFFSTORE") and len(cmd_args) >= 2:
        dest = first_key
        source_keys = [str(a) for a in cmd_args[1:]]
        return RedisAccessResult(read_keys=source_keys, write_keys=[dest], is_transaction_control=False)

    # SUNION, SINTER, SDIFF — read multiple keys, no writes.
    if upper in ("SUNION", "SINTER", "SDIFF"):
        keys = [str(a) for a in cmd_args]
        return RedisAccessResult(read_keys=keys, write_keys=[], is_transaction_control=False)

    # Sorted set store operations: ZUNIONSTORE, ZINTERSTORE, ZDIFFSTORE
    # Args: destination numkeys key [key ...]
    if upper in ("ZUNIONSTORE", "ZINTERSTORE", "ZDIFFSTORE") and len(cmd_args) >= 3:
        dest = first_key
        try:
            numkeys = int(str(cmd_args[1]))
        except (ValueError, TypeError):
            numkeys = len(cmd_args) - 2
        source_keys = [str(cmd_args[2 + i]) for i in range(min(numkeys, len(cmd_args) - 2))]
        return RedisAccessResult(read_keys=source_keys, write_keys=[dest], is_transaction_control=False)

    # SORT — can have STORE destination (write); otherwise read-only.
    if upper == "SORT":
        args_upper = [str(a).upper() for a in cmd_args]
        if "STORE" in args_upper:
            store_idx = args_upper.index("STORE")
            if store_idx + 1 < len(cmd_args):
                dest = str(cmd_args[store_idx + 1])
                return RedisAccessResult(read_keys=[first_key], write_keys=[dest], is_transaction_control=False)
        return RedisAccessResult(read_keys=[first_key], write_keys=[], is_transaction_control=False)

    # COPY — source destination
    if upper == "COPY" and len(cmd_args) >= 2:
        return RedisAccessResult(
            read_keys=[first_key],
            write_keys=[str(cmd_args[1])],
            is_transaction_control=False,
        )

    # XREAD / XREADGROUP — complex args, extract key names after STREAMS keyword.
    if upper in ("XREAD", "XREADGROUP"):
        args_upper = [str(a).upper() for a in cmd_args]
        if "STREAMS" in args_upper:
            streams_idx = args_upper.index("STREAMS")
            remaining = list(cmd_args[streams_idx + 1 :])
            # After STREAMS, args are: key1 key2 ... id1 id2 ...
            n_streams = len(remaining) // 2
            keys = [str(remaining[i]) for i in range(n_streams)]
            return RedisAccessResult(read_keys=keys, write_keys=[], is_transaction_control=False)
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)

    # XGROUP — subcommand with key.
    if upper == "XGROUP" and len(cmd_args) >= 2:
        return RedisAccessResult(read_keys=[], write_keys=[str(cmd_args[1])], is_transaction_control=False)

    # EVAL / EVALSHA — EVAL script numkeys key [key ...]
    if upper in ("EVAL", "EVALSHA", "EVAL_RO", "EVALSHA_RO") and len(cmd_args) >= 2:
        try:
            numkeys = int(str(cmd_args[1]))
        except (ValueError, TypeError):
            numkeys = 0
        keys = [str(cmd_args[2 + i]) for i in range(min(numkeys, len(cmd_args) - 2))]
        if upper.endswith("_RO"):
            return RedisAccessResult(read_keys=keys, write_keys=[], is_transaction_control=False)
        # EVAL scripts can both read and write.
        return RedisAccessResult(read_keys=keys, write_keys=keys, is_transaction_control=False)

    # PUBLISH — channel, message.
    if upper == "PUBLISH" and len(cmd_args) >= 1:
        channel = f"channel:{first_key}"
        return RedisAccessResult(read_keys=[], write_keys=[channel], is_transaction_control=False)

    # SUBSCRIBE / PSUBSCRIBE — channels.
    if upper in ("SUBSCRIBE", "PSUBSCRIBE"):
        channels = [f"channel:{a}" for a in cmd_args]
        return RedisAccessResult(read_keys=channels, write_keys=[], is_transaction_control=False)

    # GEORADIUS / GEOSEARCH — read commands with key.
    if upper in ("GEORADIUS", "GEORADIUSBYMEMBER", "GEOSEARCH", "GEODIST", "GEOPOS", "GEOHASH"):
        return RedisAccessResult(read_keys=[first_key], write_keys=[], is_transaction_control=False)

    # GEOADD / GEOSEARCHSTORE — write commands.
    if upper == "GEOADD":
        return RedisAccessResult(read_keys=[], write_keys=[first_key], is_transaction_control=False)
    if upper == "GEOSEARCHSTORE" and len(cmd_args) >= 2:
        return RedisAccessResult(
            read_keys=[str(cmd_args[1])],
            write_keys=[first_key],
            is_transaction_control=False,
        )

    # OBJECT HELP, OBJECT ENCODING, etc. — first real key is second arg.
    if upper == "OBJECT" and len(cmd_args) >= 2:
        return RedisAccessResult(read_keys=[str(cmd_args[1])], write_keys=[], is_transaction_control=False)

    # WAIT, WAITAOF — no keys.
    if upper in ("WAIT", "WAITAOF"):
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)

    # PFMERGE — destination key [key ...]
    if upper == "PFMERGE":
        source_keys = [str(a) for a in cmd_args[1:]]
        return RedisAccessResult(read_keys=source_keys, write_keys=[first_key], is_transaction_control=False)

    # BITOP — operation destkey key [key ...]
    if upper == "BITOP" and len(cmd_args) >= 3:
        dest = str(cmd_args[1])
        source_keys = [str(a) for a in cmd_args[2:]]
        return RedisAccessResult(read_keys=source_keys, write_keys=[dest], is_transaction_control=False)

    # BITCOUNT, BITPOS, BITFIELD_RO — read single key.
    if upper in ("BITCOUNT", "BITPOS", "BITFIELD_RO", "GETBIT"):
        return RedisAccessResult(read_keys=[first_key], write_keys=[], is_transaction_control=False)

    # BITFIELD, SETBIT — write single key.
    if upper in ("BITFIELD", "SETBIT"):
        return RedisAccessResult(read_keys=[], write_keys=[first_key], is_transaction_control=False)

    # SCAN, DBSIZE, FLUSHDB, SELECT, etc. — server commands, no specific key conflicts.
    # KEYS, RANDOMKEY — server-level scan, not key-specific.
    if upper in (
        "PING",
        "ECHO",
        "INFO",
        "CONFIG",
        "CLIENT",
        "SLOWLOG",
        "DEBUG",
        "DBSIZE",
        "FLUSHDB",
        "FLUSHALL",
        "SELECT",
        "SWAPDB",
        "AUTH",
        "SCAN",
        "KEYS",
        "RANDOMKEY",
        "WAIT",
        "COMMAND",
        "TIME",
        "CLUSTER",
        "READONLY",
        "READWRITE",
        "ASKING",
        "LATENCY",
        "MEMORY",
        "MODULE",
        "ACL",
        "RESET",
        "QUIT",
        "SHUTDOWN",
        "BGSAVE",
        "BGREWRITEAOF",
        "SAVE",
        "LASTSAVE",
        "UNSUBSCRIBE",
        "PUNSUBSCRIBE",
    ):
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)

    # Fallback: treat first argument as a write key (conservative).
    return RedisAccessResult(read_keys=[], write_keys=[first_key], is_transaction_control=False)
