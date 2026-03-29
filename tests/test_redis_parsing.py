"""Tests for Redis command classification and key extraction."""

from __future__ import annotations

import pytest

from frontrun._redis_parsing import parse_redis_access


class TestStringCommands:
    """Test classification of Redis string commands."""

    def test_get_is_read(self) -> None:
        result = parse_redis_access("GET", ("mykey",))
        assert result.read_keys == ["mykey"]
        assert result.write_keys == []

    def test_set_is_write(self) -> None:
        result = parse_redis_access("SET", ("mykey", "myvalue"))
        assert result.read_keys == []
        assert result.write_keys == ["mykey"]

    def test_setnx_is_write(self) -> None:
        result = parse_redis_access("SETNX", ("mykey", "myvalue"))
        assert result.read_keys == []
        assert result.write_keys == ["mykey"]

    def test_setex_is_write(self) -> None:
        result = parse_redis_access("SETEX", ("mykey", 60, "myvalue"))
        assert result.read_keys == []
        assert result.write_keys == ["mykey"]

    def test_incr_is_write(self) -> None:
        result = parse_redis_access("INCR", ("counter",))
        assert result.read_keys == []
        assert result.write_keys == ["counter"]

    def test_incrby_is_write(self) -> None:
        result = parse_redis_access("INCRBY", ("counter", 5))
        assert result.read_keys == []
        assert result.write_keys == ["counter"]

    def test_decr_is_write(self) -> None:
        result = parse_redis_access("DECR", ("counter",))
        assert result.read_keys == []
        assert result.write_keys == ["counter"]

    def test_append_is_write(self) -> None:
        result = parse_redis_access("APPEND", ("mykey", "extra"))
        assert result.read_keys == []
        assert result.write_keys == ["mykey"]

    def test_strlen_is_read(self) -> None:
        result = parse_redis_access("STRLEN", ("mykey",))
        assert result.read_keys == ["mykey"]
        assert result.write_keys == []

    def test_mget_reads_all_keys(self) -> None:
        result = parse_redis_access("MGET", ("key1", "key2", "key3"))
        assert result.read_keys == ["key1", "key2", "key3"]
        assert result.write_keys == []

    def test_mset_writes_key_value_pairs(self) -> None:
        result = parse_redis_access("MSET", ("key1", "val1", "key2", "val2"))
        assert result.read_keys == []
        assert result.write_keys == ["key1", "key2"]

    def test_getset_is_read_write(self) -> None:
        result = parse_redis_access("GETSET", ("mykey", "newvalue"))
        assert result.read_keys == ["mykey"]
        assert result.write_keys == ["mykey"]

    def test_getdel_is_read_write(self) -> None:
        result = parse_redis_access("GETDEL", ("mykey",))
        assert result.read_keys == ["mykey"]
        assert result.write_keys == ["mykey"]


class TestHashCommands:
    """Test classification of Redis hash commands."""

    def test_hget_is_read(self) -> None:
        result = parse_redis_access("HGET", ("myhash", "field1"))
        assert result.read_keys == ["myhash"]
        assert result.write_keys == []

    def test_hset_is_write(self) -> None:
        result = parse_redis_access("HSET", ("myhash", "field1", "value1"))
        assert result.read_keys == []
        assert result.write_keys == ["myhash"]

    def test_hmset_is_write(self) -> None:
        result = parse_redis_access("HMSET", ("myhash", "f1", "v1", "f2", "v2"))
        assert result.read_keys == []
        assert result.write_keys == ["myhash"]

    def test_hgetall_is_read(self) -> None:
        result = parse_redis_access("HGETALL", ("myhash",))
        assert result.read_keys == ["myhash"]
        assert result.write_keys == []

    def test_hdel_is_write(self) -> None:
        result = parse_redis_access("HDEL", ("myhash", "field1"))
        assert result.read_keys == []
        assert result.write_keys == ["myhash"]

    def test_hincrby_is_write(self) -> None:
        result = parse_redis_access("HINCRBY", ("myhash", "field1", 5))
        assert result.read_keys == []
        assert result.write_keys == ["myhash"]

    def test_hexists_is_read(self) -> None:
        result = parse_redis_access("HEXISTS", ("myhash", "field1"))
        assert result.read_keys == ["myhash"]
        assert result.write_keys == []

    def test_hlen_is_read(self) -> None:
        result = parse_redis_access("HLEN", ("myhash",))
        assert result.read_keys == ["myhash"]
        assert result.write_keys == []


class TestListCommands:
    """Test classification of Redis list commands."""

    def test_lpush_is_write(self) -> None:
        result = parse_redis_access("LPUSH", ("mylist", "value1"))
        assert result.write_keys == ["mylist"]

    def test_rpush_is_write(self) -> None:
        result = parse_redis_access("RPUSH", ("mylist", "value1"))
        assert result.write_keys == ["mylist"]

    def test_lpop_is_write(self) -> None:
        result = parse_redis_access("LPOP", ("mylist",))
        assert result.write_keys == ["mylist"]

    def test_rpop_is_write(self) -> None:
        result = parse_redis_access("RPOP", ("mylist",))
        assert result.write_keys == ["mylist"]

    def test_lrange_is_read(self) -> None:
        result = parse_redis_access("LRANGE", ("mylist", 0, -1))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == []

    def test_llen_is_read(self) -> None:
        result = parse_redis_access("LLEN", ("mylist",))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == []

    def test_lindex_is_read(self) -> None:
        result = parse_redis_access("LINDEX", ("mylist", 0))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == []

    def test_lmpop_uses_numkeys(self) -> None:
        """LMPOP numkeys key [key ...] LEFT|RIGHT — first arg is numkeys, not a key."""
        result = parse_redis_access("LMPOP", (2, "list1", "list2", "LEFT"))
        assert result.write_keys == ["list1", "list2"]
        assert result.read_keys == ["list1", "list2"]

    def test_lmpop_single_key(self) -> None:
        result = parse_redis_access("LMPOP", (1, "mylist", "LEFT"))
        assert result.write_keys == ["mylist"]
        assert result.read_keys == ["mylist"]

    def test_blpop_is_read_write(self) -> None:
        """BLPOP pops (removes) and returns the value — both read and write."""
        result = parse_redis_access("BLPOP", ("mylist", 0))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == ["mylist"]

    def test_brpop_is_read_write(self) -> None:
        """BRPOP pops (removes) and returns the value — both read and write."""
        result = parse_redis_access("BRPOP", ("mylist", 0))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == ["mylist"]

    def test_blpop_multiple_keys(self) -> None:
        """BLPOP key [key ...] timeout — all keys are read+write."""
        result = parse_redis_access("BLPOP", ("list1", "list2", 0))
        assert result.read_keys == ["list1", "list2"]
        assert result.write_keys == ["list1", "list2"]

    def test_brpop_multiple_keys(self) -> None:
        """BRPOP key [key ...] timeout — all keys are read+write."""
        result = parse_redis_access("BRPOP", ("list1", "list2", 0))
        assert result.read_keys == ["list1", "list2"]
        assert result.write_keys == ["list1", "list2"]


class TestSetCommands:
    """Test classification of Redis set commands."""

    def test_sadd_is_write(self) -> None:
        result = parse_redis_access("SADD", ("myset", "member1"))
        assert result.write_keys == ["myset"]

    def test_srem_is_write(self) -> None:
        result = parse_redis_access("SREM", ("myset", "member1"))
        assert result.write_keys == ["myset"]

    def test_smembers_is_read(self) -> None:
        result = parse_redis_access("SMEMBERS", ("myset",))
        assert result.read_keys == ["myset"]

    def test_scard_is_read(self) -> None:
        result = parse_redis_access("SCARD", ("myset",))
        assert result.read_keys == ["myset"]

    def test_sismember_is_read(self) -> None:
        result = parse_redis_access("SISMEMBER", ("myset", "member1"))
        assert result.read_keys == ["myset"]

    def test_sunionstore_reads_sources_writes_dest(self) -> None:
        result = parse_redis_access("SUNIONSTORE", ("dest", "src1", "src2"))
        assert result.read_keys == ["src1", "src2"]
        assert result.write_keys == ["dest"]

    def test_sunion_reads_all(self) -> None:
        result = parse_redis_access("SUNION", ("set1", "set2", "set3"))
        assert result.read_keys == ["set1", "set2", "set3"]
        assert result.write_keys == []


class TestSortedSetCommands:
    """Test classification of Redis sorted set commands."""

    def test_zadd_is_write(self) -> None:
        result = parse_redis_access("ZADD", ("myzset", 1, "member1"))
        assert result.write_keys == ["myzset"]

    def test_zrange_is_read(self) -> None:
        result = parse_redis_access("ZRANGE", ("myzset", 0, -1))
        assert result.read_keys == ["myzset"]

    def test_zcard_is_read(self) -> None:
        result = parse_redis_access("ZCARD", ("myzset",))
        assert result.read_keys == ["myzset"]

    def test_zincrby_is_write(self) -> None:
        result = parse_redis_access("ZINCRBY", ("myzset", 1, "member1"))
        assert result.write_keys == ["myzset"]

    def test_zmpop_uses_numkeys(self) -> None:
        """ZMPOP numkeys key [key ...] MIN|MAX — first arg is numkeys."""
        result = parse_redis_access("ZMPOP", (2, "zset1", "zset2", "MIN"))
        assert result.write_keys == ["zset1", "zset2"]
        assert result.read_keys == ["zset1", "zset2"]

    def test_bzmpop_uses_numkeys(self) -> None:
        """BZMPOP timeout numkeys key [key ...] MIN|MAX — second arg is numkeys."""
        result = parse_redis_access("BZMPOP", (0, 2, "zset1", "zset2", "MIN"))
        assert result.write_keys == ["zset1", "zset2"]
        assert result.read_keys == ["zset1", "zset2"]

    def test_bzpopmin_is_read_write(self) -> None:
        """BZPOPMIN pops (removes) and returns the min element — both read and write."""
        result = parse_redis_access("BZPOPMIN", ("myzset", 0))
        assert result.read_keys == ["myzset"]
        assert result.write_keys == ["myzset"]

    def test_bzpopmax_is_read_write(self) -> None:
        """BZPOPMAX pops (removes) and returns the max element — both read and write."""
        result = parse_redis_access("BZPOPMAX", ("myzset", 0))
        assert result.read_keys == ["myzset"]
        assert result.write_keys == ["myzset"]


class TestKeyCommands:
    """Test classification of Redis key commands."""

    def test_del_writes_all_keys(self) -> None:
        result = parse_redis_access("DEL", ("key1", "key2"))
        assert result.read_keys == []
        assert result.write_keys == ["key1", "key2"]

    def test_exists_reads_all_keys(self) -> None:
        result = parse_redis_access("EXISTS", ("key1", "key2"))
        assert result.read_keys == ["key1", "key2"]
        assert result.write_keys == []

    def test_expire_is_write(self) -> None:
        result = parse_redis_access("EXPIRE", ("mykey", 60))
        assert result.write_keys == ["mykey"]

    def test_ttl_is_read(self) -> None:
        result = parse_redis_access("TTL", ("mykey",))
        assert result.read_keys == ["mykey"]
        assert result.write_keys == []

    def test_type_is_read(self) -> None:
        result = parse_redis_access("TYPE", ("mykey",))
        assert result.read_keys == ["mykey"]
        assert result.write_keys == []

    def test_rename_reads_source_writes_both(self) -> None:
        result = parse_redis_access("RENAME", ("old", "new"))
        assert result.read_keys == ["old"]
        assert result.write_keys == ["old", "new"]

    def test_persist_is_read_write(self) -> None:
        """PERSIST modifies TTL metadata — should be read+write."""
        result = parse_redis_access("PERSIST", ("mykey",))
        assert result.read_keys == ["mykey"]
        assert result.write_keys == ["mykey"]

    def test_getex_is_read_write(self) -> None:
        """GETEX reads value and may modify TTL — should be read+write."""
        result = parse_redis_access("GETEX", ("mykey",))
        assert result.read_keys == ["mykey"]
        assert result.write_keys == ["mykey"]


class TestTransactionCommands:
    """Test classification of Redis transaction commands."""

    def test_multi_is_tx_control(self) -> None:
        result = parse_redis_access("MULTI", ())
        assert result.is_transaction_control is True
        assert result.read_keys == []
        assert result.write_keys == []

    def test_exec_is_tx_control(self) -> None:
        result = parse_redis_access("EXEC", ())
        assert result.is_transaction_control is True

    def test_discard_is_tx_control(self) -> None:
        result = parse_redis_access("DISCARD", ())
        assert result.is_transaction_control is True

    def test_watch_reads_keys(self) -> None:
        result = parse_redis_access("WATCH", ("key1", "key2"))
        assert result.is_transaction_control is True
        assert result.read_keys == ["key1", "key2"]


class TestSpecialCommands:
    """Test classification of special Redis commands."""

    def test_rpoplpush_reads_source_writes_both(self) -> None:
        result = parse_redis_access("RPOPLPUSH", ("source", "dest"))
        assert result.read_keys == ["source"]
        assert result.write_keys == ["source", "dest"]

    def test_sort_with_store(self) -> None:
        result = parse_redis_access("SORT", ("mylist", "STORE", "dest"))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == ["dest"]

    def test_sort_without_store(self) -> None:
        result = parse_redis_access("SORT", ("mylist",))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == []

    def test_copy_reads_source_writes_dest(self) -> None:
        result = parse_redis_access("COPY", ("source", "dest"))
        assert result.read_keys == ["source"]
        assert result.write_keys == ["dest"]

    def test_eval_is_atomic(self) -> None:
        # EVAL/EVALSHA are atomic (Lua scripts execute without interleaving).
        # They are treated as transaction control with no key-level accesses.
        result = parse_redis_access("EVAL", ("script", 2, "key1", "key2", "arg1"))
        assert result.read_keys == []
        assert result.write_keys == []
        assert result.is_transaction_control

    def test_pfadd_is_write(self) -> None:
        result = parse_redis_access("PFADD", ("myhll", "elem1"))
        assert result.write_keys == ["myhll"]

    def test_pfcount_is_read(self) -> None:
        result = parse_redis_access("PFCOUNT", ("myhll",))
        assert result.read_keys == ["myhll"]

    def test_pfmerge_reads_sources_writes_dest(self) -> None:
        result = parse_redis_access("PFMERGE", ("dest", "src1", "src2"))
        assert result.read_keys == ["src1", "src2"]
        assert result.write_keys == ["dest"]

    def test_bitop_reads_sources_writes_dest(self) -> None:
        result = parse_redis_access("BITOP", ("AND", "dest", "src1", "src2"))
        assert result.read_keys == ["src1", "src2"]
        assert result.write_keys == ["dest"]


class TestServerCommands:
    """Test that server commands produce no key conflicts."""

    @pytest.mark.parametrize("cmd", ["PING", "INFO", "DBSIZE", "FLUSHDB", "SELECT", "AUTH"])
    def test_server_commands_no_keys(self, cmd: str) -> None:
        result = parse_redis_access(cmd, ())
        assert result.read_keys == []
        assert result.write_keys == []

    def test_ping_with_message(self) -> None:
        # PING is in the server commands set, so even with args it returns no keys
        result = parse_redis_access("PING", ("hello",))
        assert result.read_keys == []
        assert result.write_keys == []


class TestCaseInsensitivity:
    """Test that command names are case-insensitive."""

    def test_lowercase_get(self) -> None:
        result = parse_redis_access("get", ("mykey",))
        assert result.read_keys == ["mykey"]

    def test_mixed_case_set(self) -> None:
        result = parse_redis_access("Set", ("mykey", "val"))
        assert result.write_keys == ["mykey"]


class TestStreamCommands:
    """Test classification of Redis stream commands."""

    def test_xadd_is_write(self) -> None:
        result = parse_redis_access("XADD", ("mystream", "*", "field1", "value1"))
        assert result.write_keys == ["mystream"]

    def test_xlen_is_read(self) -> None:
        result = parse_redis_access("XLEN", ("mystream",))
        assert result.read_keys == ["mystream"]

    def test_xrange_is_read(self) -> None:
        result = parse_redis_access("XRANGE", ("mystream", "-", "+"))
        assert result.read_keys == ["mystream"]

    def test_xread_extracts_stream_keys(self) -> None:
        result = parse_redis_access("XREAD", ("COUNT", "10", "STREAMS", "stream1", "stream2", "0", "0"))
        assert result.read_keys == ["stream1", "stream2"]

    def test_xdel_is_write(self) -> None:
        result = parse_redis_access("XDEL", ("mystream", "id1"))
        assert result.write_keys == ["mystream"]

    def test_xreadgroup_is_read_write(self) -> None:
        """XREADGROUP advances consumer group state — should be read+write."""
        result = parse_redis_access("XREADGROUP", ("GROUP", "mygroup", "consumer1", "STREAMS", "stream1", ">"))
        assert result.read_keys == ["stream1"]
        assert result.write_keys == ["stream1"]

    def test_xread_is_read_only(self) -> None:
        """XREAD is pure read (no consumer group state mutation)."""
        result = parse_redis_access("XREAD", ("STREAMS", "stream1", "0"))
        assert result.read_keys == ["stream1"]
        assert result.write_keys == []


class TestGeoCommands:
    """Test classification of Redis geo commands."""

    def test_georadius_read_only(self) -> None:
        result = parse_redis_access("GEORADIUS", ("geo", 15, 37, 100, "km"))
        assert result.read_keys == ["geo"]
        assert result.write_keys == []

    def test_georadius_with_store(self) -> None:
        """GEORADIUS with STORE writes a destination key."""
        result = parse_redis_access("GEORADIUS", ("geo", 15, 37, 100, "km", "STORE", "dest"))
        assert result.read_keys == ["geo"]
        assert result.write_keys == ["dest"]

    def test_georadius_with_storedist(self) -> None:
        """GEORADIUS with STOREDIST writes a destination key."""
        result = parse_redis_access("GEORADIUS", ("geo", 15, 37, 100, "km", "STOREDIST", "dest"))
        assert result.read_keys == ["geo"]
        assert result.write_keys == ["dest"]

    def test_georadiusbymember_with_store(self) -> None:
        result = parse_redis_access("GEORADIUSBYMEMBER", ("geo", "member", 100, "km", "STORE", "dest"))
        assert result.read_keys == ["geo"]
        assert result.write_keys == ["dest"]


class TestCommandCoverage:
    """Verify that all core Redis commands are handled (not falling through to fallback)."""

    # Complete list of core Redis commands (from redis.io/commands, excluding
    # module commands like BF.*, CF.*, FT.*, JSON.*, TS.*, TDIGEST.*, TOPK.*, CMS.*,
    # and internal/debug commands like PSYNC, REPLCONF, SYNC, RESTORE-ASKING, etc.)
    # Also excludes sub-commands (CLIENT LIST, CONFIG GET, etc.) since redis-py
    # sends these as the base command name.
    CORE_COMMANDS: list[tuple[str, tuple[object, ...]]] = [
        # String commands
        ("GET", ("k",)),
        ("SET", ("k", "v")),
        ("SETNX", ("k", "v")),
        ("SETEX", ("k", 60, "v")),
        ("PSETEX", ("k", 60000, "v")),
        ("SETRANGE", ("k", 0, "v")),
        ("GETRANGE", ("k", 0, 10)),
        ("GETSET", ("k", "v")),
        ("GETDEL", ("k",)),
        ("GETEX", ("k",)),
        ("MGET", ("k1", "k2")),
        ("MSET", ("k1", "v1", "k2", "v2")),
        ("MSETNX", ("k1", "v1")),
        ("APPEND", ("k", "v")),
        ("STRLEN", ("k",)),
        ("INCR", ("k",)),
        ("INCRBY", ("k", 5)),
        ("INCRBYFLOAT", ("k", 1.5)),
        ("DECR", ("k",)),
        ("DECRBY", ("k", 5)),
        ("SUBSTR", ("k", 0, 5)),
        # Hash commands
        ("HGET", ("h", "f")),
        ("HSET", ("h", "f", "v")),
        ("HSETNX", ("h", "f", "v")),
        ("HMSET", ("h", "f", "v")),
        ("HMGET", ("h", "f1", "f2")),
        ("HGETALL", ("h",)),
        ("HDEL", ("h", "f")),
        ("HEXISTS", ("h", "f")),
        ("HINCRBY", ("h", "f", 5)),
        ("HINCRBYFLOAT", ("h", "f", 1.5)),
        ("HKEYS", ("h",)),
        ("HVALS", ("h",)),
        ("HLEN", ("h",)),
        ("HRANDFIELD", ("h",)),
        ("HSCAN", ("h", 0)),
        # List commands
        ("LPUSH", ("l", "v")),
        ("LPUSHX", ("l", "v")),
        ("RPUSH", ("l", "v")),
        ("RPUSHX", ("l", "v")),
        ("LPOP", ("l",)),
        ("RPOP", ("l",)),
        ("LLEN", ("l",)),
        ("LRANGE", ("l", 0, -1)),
        ("LINDEX", ("l", 0)),
        ("LSET", ("l", 0, "v")),
        ("LINSERT", ("l", "BEFORE", "pivot", "v")),
        ("LREM", ("l", 0, "v")),
        ("LTRIM", ("l", 0, 10)),
        ("LPOS", ("l", "v")),
        ("LMOVE", ("src", "dst", "LEFT", "RIGHT")),
        ("BLMOVE", ("src", "dst", "LEFT", "RIGHT", 0)),
        ("LMPOP", (1, "l", "LEFT")),
        ("BLPOP", ("l", 0)),
        ("BRPOP", ("l", 0)),
        ("RPOPLPUSH", ("src", "dst")),
        ("BRPOPLPUSH", ("src", "dst", 0)),
        # Set commands
        ("SADD", ("s", "m")),
        ("SREM", ("s", "m")),
        ("SPOP", ("s",)),
        ("SCARD", ("s",)),
        ("SISMEMBER", ("s", "m")),
        ("SMISMEMBER", ("s", "m1", "m2")),
        ("SMEMBERS", ("s",)),
        ("SRANDMEMBER", ("s",)),
        ("SSCAN", ("s", 0)),
        ("SMOVE", ("src", "dst", "m")),
        ("SUNION", ("s1", "s2")),
        ("SUNIONSTORE", ("dst", "s1", "s2")),
        ("SINTER", ("s1", "s2")),
        ("SINTERSTORE", ("dst", "s1", "s2")),
        ("SDIFF", ("s1", "s2")),
        ("SDIFFSTORE", ("dst", "s1", "s2")),
        # Sorted set commands
        ("ZADD", ("z", 1, "m")),
        ("ZREM", ("z", "m")),
        ("ZINCRBY", ("z", 1, "m")),
        ("ZCARD", ("z",)),
        ("ZCOUNT", ("z", "-inf", "+inf")),
        ("ZLEXCOUNT", ("z", "-", "+")),
        ("ZRANGE", ("z", 0, -1)),
        ("ZRANGEBYLEX", ("z", "-", "+")),
        ("ZRANGEBYSCORE", ("z", "-inf", "+inf")),
        ("ZREVRANGE", ("z", 0, -1)),
        ("ZREVRANGEBYLEX", ("z", "+", "-")),
        ("ZREVRANGEBYSCORE", ("z", "+inf", "-inf")),
        ("ZRANK", ("z", "m")),
        ("ZREVRANK", ("z", "m")),
        ("ZSCORE", ("z", "m")),
        ("ZMSCORE", ("z", "m1", "m2")),
        ("ZPOPMIN", ("z",)),
        ("ZPOPMAX", ("z",)),
        ("BZPOPMIN", ("z", 0)),
        ("BZPOPMAX", ("z", 0)),
        ("ZRANDMEMBER", ("z",)),
        ("ZSCAN", ("z", 0)),
        ("ZREMRANGEBYLEX", ("z", "-", "+")),
        ("ZREMRANGEBYRANK", ("z", 0, 1)),
        ("ZREMRANGEBYSCORE", ("z", 0, 100)),
        ("ZUNIONSTORE", ("dst", 2, "z1", "z2")),
        ("ZINTERSTORE", ("dst", 2, "z1", "z2")),
        ("ZDIFFSTORE", ("dst", 2, "z1", "z2")),
        ("ZMPOP", (1, "z", "MIN")),
        ("BZMPOP", (0, 1, "z", "MIN")),
        # Key commands
        ("DEL", ("k",)),
        ("UNLINK", ("k",)),
        ("EXISTS", ("k",)),
        ("EXPIRE", ("k", 60)),
        ("EXPIREAT", ("k", 1000000)),
        ("PEXPIRE", ("k", 60000)),
        ("PEXPIREAT", ("k", 1000000000)),
        ("EXPIRETIME", ("k",)),
        ("PEXPIRETIME", ("k",)),
        ("TTL", ("k",)),
        ("PTTL", ("k",)),
        ("PERSIST", ("k",)),
        ("TYPE", ("k",)),
        ("DUMP", ("k",)),
        ("RENAME", ("old", "new")),
        ("RENAMENX", ("old", "new")),
        ("COPY", ("src", "dst")),
        ("SORT", ("k",)),
        # HyperLogLog
        ("PFADD", ("hll", "e")),
        ("PFCOUNT", ("hll",)),
        ("PFMERGE", ("dst", "src1", "src2")),
        # Pub/Sub
        ("PUBLISH", ("chan", "msg")),
        ("SUBSCRIBE", ("chan",)),
        ("PSUBSCRIBE", ("pattern",)),
        # Streams
        ("XADD", ("s", "*", "f", "v")),
        ("XDEL", ("s", "id")),
        ("XLEN", ("s",)),
        ("XRANGE", ("s", "-", "+")),
        ("XREVRANGE", ("s", "+", "-")),
        ("XREAD", ("STREAMS", "s1", "0")),
        ("XREADGROUP", ("GROUP", "g", "c", "STREAMS", "s1", ">")),
        ("XINFO", ("s",)),
        ("XPENDING", ("s", "g")),
        ("XTRIM", ("s", "MAXLEN", 100)),
        ("XACK", ("s", "g", "id")),
        ("XGROUP", ("CREATE", "s", "g", "0")),
        # Geo
        ("GEOADD", ("g", 13.361, 38.116, "m")),
        ("GEODIST", ("g", "m1", "m2")),
        ("GEOHASH", ("g", "m")),
        ("GEOPOS", ("g", "m")),
        ("GEORADIUS", ("g", 15, 37, 100, "km")),
        ("GEORADIUSBYMEMBER", ("g", "m", 100, "km")),
        ("GEOSEARCH", ("g", "FROMMEMBER", "m", "BYRADIUS", 100, "km")),
        ("GEOSEARCHSTORE", ("dst", "g", "FROMMEMBER", "m", "BYRADIUS", 100, "km")),
        # Bit commands
        ("BITCOUNT", ("k",)),
        ("BITPOS", ("k", 1)),
        ("BITOP", ("AND", "dst", "k1", "k2")),
        ("BITFIELD", ("k", "SET", "u8", 0, 100)),
        ("BITFIELD_RO", ("k", "GET", "u8", 0)),
        ("GETBIT", ("k", 0)),
        ("SETBIT", ("k", 0, 1)),
        # Scripting
        ("EVAL", ("script", 1, "k1")),
        ("EVALSHA", ("sha", 1, "k1")),
        ("EVAL_RO", ("script", 1, "k1")),
        ("EVALSHA_RO", ("sha", 1, "k1")),
        # Transaction
        ("MULTI", ()),
        ("EXEC", ()),
        ("DISCARD", ()),
        ("WATCH", ("k1", "k2")),
        ("UNWATCH", ()),
        # Server / connection (no keys)
        ("PING", ()),
        ("ECHO", ("msg",)),
        ("INFO", ()),
        ("SELECT", ("0",)),
        ("AUTH", ("password",)),
        ("DBSIZE", ()),
        ("FLUSHDB", ()),
        ("FLUSHALL", ()),
        ("SAVE", ()),
        ("BGSAVE", ()),
        ("BGREWRITEAOF", ()),
        ("LASTSAVE", ()),
        ("SHUTDOWN", ()),
        ("QUIT", ()),
        ("RESET", ()),
        ("WAIT", ("1", "0")),
        ("WAITAOF", ("1", "1", "0")),
        ("SCAN", ("0",)),
        ("KEYS", ("*",)),
        ("RANDOMKEY", ()),
        ("TIME", ()),
        ("COMMAND", ()),
        ("SWAPDB", ("0", "1")),
        # Cluster
        ("CLUSTER", ("INFO",)),
        ("READONLY", ()),
        ("READWRITE", ()),
        ("ASKING", ()),
        # Slow/config/client (base commands — sub-commands handled by redis-py)
        ("SLOWLOG", ("GET",)),
        ("CONFIG", ("GET", "maxmemory")),
        ("CLIENT", ("LIST",)),
        ("LATENCY", ("LATEST",)),
        ("MEMORY", ("USAGE", "k")),
        ("MODULE", ("LIST",)),
        ("ACL", ("LIST",)),
        ("OBJECT", ("ENCODING", "k")),
        ("DEBUG", ("OBJECT", "k")),  # Server command — no keys
        ("UNSUBSCRIBE", ()),
        ("PUNSUBSCRIBE", ()),
    ]

    def test_all_core_commands_are_handled(self) -> None:
        """Every core Redis command should produce a non-fallback classification.

        The fallback is write-only on first_key with no read_keys. We verify that
        every command either:
        - Has explicit read/write keys, OR
        - Is a server/transaction command with no keys, OR
        - Falls into a known "no keys" category
        """
        # Commands that legitimately return no keys (server, connection, cluster)
        no_key_commands = {
            "PING",
            "ECHO",
            "INFO",
            "SELECT",
            "AUTH",
            "DBSIZE",
            "FLUSHDB",
            "FLUSHALL",
            "SAVE",
            "BGSAVE",
            "BGREWRITEAOF",
            "LASTSAVE",
            "SHUTDOWN",
            "QUIT",
            "RESET",
            "SCAN",
            "KEYS",
            "RANDOMKEY",
            "TIME",
            "COMMAND",
            "SWAPDB",
            "CLUSTER",
            "READONLY",
            "READWRITE",
            "ASKING",
            "SLOWLOG",
            "CONFIG",
            "CLIENT",
            "LATENCY",
            "MEMORY",
            "MODULE",
            "ACL",
            "WAIT",
            "WAITAOF",
            "UNSUBSCRIBE",
            "PUNSUBSCRIBE",
            "MULTI",
            "EXEC",
            "DISCARD",
            "UNWATCH",
            "DEBUG",
            # EVAL/EVALSHA are treated as atomic (transaction control)
            # with no key-level accesses.  See defect #8.
            "EVAL",
            "EVALSHA",
            "EVAL_RO",
            "EVALSHA_RO",
        }

        for cmd_name, cmd_args in self.CORE_COMMANDS:
            result = parse_redis_access(cmd_name, cmd_args)
            upper = cmd_name.upper()

            if upper in no_key_commands:
                # These should have no keys (that's correct)
                continue

            has_keys = bool(result.read_keys or result.write_keys)
            assert has_keys, (
                f"Command {cmd_name} with args {cmd_args} returned no keys — "
                f"likely hitting the fallback. Add explicit handling."
            )

    def test_no_command_in_multiple_single_key_sets(self) -> None:
        """Verify no command appears in both read and write single-key sets."""
        from frontrun._redis_parsing import (
            _SINGLE_KEY_READ_CMDS,
            _SINGLE_KEY_READ_WRITE_CMDS,
            _SINGLE_KEY_WRITE_CMDS,
        )

        overlap_rw = _SINGLE_KEY_READ_CMDS & _SINGLE_KEY_WRITE_CMDS
        assert not overlap_rw, f"Commands in both read and write sets: {overlap_rw}"

        overlap_r_rw = _SINGLE_KEY_READ_CMDS & _SINGLE_KEY_READ_WRITE_CMDS
        assert not overlap_r_rw, f"Commands in both read and read+write sets: {overlap_r_rw}"

        overlap_w_rw = _SINGLE_KEY_WRITE_CMDS & _SINGLE_KEY_READ_WRITE_CMDS
        assert not overlap_w_rw, f"Commands in both write and read+write sets: {overlap_w_rw}"


class TestConflictDetection:
    """Verify that commands on the same key produce overlapping resource sets
    and commands on different keys produce disjoint resource sets."""

    def test_same_key_read_write_conflicts(self) -> None:
        """GET and SET on the same key should produce overlapping key sets."""
        r1 = parse_redis_access("GET", ("shared",))
        r2 = parse_redis_access("SET", ("shared", "val"))
        assert set(r1.read_keys) & set(r2.write_keys), "GET and SET on same key must conflict"

    def test_same_key_write_write_conflicts(self) -> None:
        """Two SETs on the same key should produce overlapping write sets."""
        r1 = parse_redis_access("SET", ("shared", "v1"))
        r2 = parse_redis_access("SET", ("shared", "v2"))
        assert set(r1.write_keys) & set(r2.write_keys), "Two SETs on same key must conflict"

    def test_different_keys_no_conflict(self) -> None:
        """Operations on different keys should produce disjoint key sets."""
        r1 = parse_redis_access("SET", ("key_a", "v"))
        r2 = parse_redis_access("SET", ("key_b", "v"))
        all_1 = set(r1.read_keys + r1.write_keys)
        all_2 = set(r2.read_keys + r2.write_keys)
        assert not (all_1 & all_2), "Operations on different keys must not conflict"

    def test_many_independent_ops_no_overlap(self) -> None:
        """A batch of operations on distinct keys should have no pairwise overlap."""
        ops = [parse_redis_access("SET", (f"ind:{i}", "v")) for i in range(20)]
        for i, r1 in enumerate(ops):
            for j, r2 in enumerate(ops):
                if i == j:
                    continue
                all_1 = set(r1.read_keys + r1.write_keys)
                all_2 = set(r2.read_keys + r2.write_keys)
                assert not (all_1 & all_2), f"Ops {i} and {j} should not conflict"

    def test_mixed_commands_same_key_all_conflict(self) -> None:
        """Various read/write commands on the same key should all conflict pairwise."""
        cmds = [
            ("GET", ("k",)),
            ("SET", ("k", "v")),
            ("INCR", ("k",)),
            ("APPEND", ("k", "x")),
            ("HSET", ("k", "f", "v")),
            ("HGET", ("k", "f")),
            ("LPUSH", ("k", "v")),
            ("SADD", ("k", "m")),
        ]
        for i, (c1, a1) in enumerate(cmds):
            for j, (c2, a2) in enumerate(cmds):
                if i == j:
                    continue
                r1 = parse_redis_access(c1, a1)
                r2 = parse_redis_access(c2, a2)
                all_1 = set(r1.read_keys + r1.write_keys)
                all_2 = set(r2.read_keys + r2.write_keys)
                assert all_1 & all_2, f"{c1} and {c2} on same key must conflict"


class TestResourceIdConstruction:
    """Verify resource ID construction for different db scopes."""

    def test_resource_id_without_scope(self) -> None:
        from frontrun._redis_client import _redis_resource_id

        assert _redis_resource_id("mykey") == "redis:mykey"

    def test_resource_id_with_scope(self) -> None:
        from frontrun._redis_client import _redis_resource_id

        assert _redis_resource_id("mykey", db_scope="redis:localhost:6379/0") == (
            "redis:mykey:db=redis:localhost:6379/0"
        )

    def test_different_db_scopes_produce_different_ids(self) -> None:
        from frontrun._redis_client import _redis_resource_id

        id1 = _redis_resource_id("mykey", db_scope="redis:localhost:6379/0")
        id2 = _redis_resource_id("mykey", db_scope="redis:localhost:6379/1")
        assert id1 != id2, "Same key on different DBs must produce different resource IDs"

    def test_same_db_scope_same_key_produces_same_id(self) -> None:
        from frontrun._redis_client import _redis_resource_id

        id1 = _redis_resource_id("mykey", db_scope="redis:localhost:6379/0")
        id2 = _redis_resource_id("mykey", db_scope="redis:localhost:6379/0")
        assert id1 == id2


class TestDporEngineIoIndependence:
    """Verify the Rust DPOR engine correctly handles I/O independence.

    These tests directly exercise the engine without opcode tracing, proving
    that independent I/O objects (different Redis keys mapped to different
    object IDs) result in exactly 1 execution path, while conflicting objects
    require additional exploration.
    """

    def test_independent_io_objects_single_path(self) -> None:
        """Two threads writing different I/O objects → exactly 1 DPOR path."""
        from frontrun._dpor import PyDporEngine

        engine = PyDporEngine(num_threads=2, preemption_bound=2, max_branches=100)
        execution = engine.begin_execution()

        # Schedule: thread 0 runs first
        engine.schedule(execution)
        engine.report_io_access(execution, 0, 100, "write")

        # Schedule again
        engine.schedule(execution)
        engine.report_io_access(execution, 1, 200, "write")  # Different object!

        execution.finish_thread(0)
        engine.schedule(execution)
        execution.finish_thread(1)

        assert not engine.next_execution(), "Independent I/O objects should need exactly 1 path"

    def test_conflicting_io_objects_multiple_paths(self) -> None:
        """Two threads writing the SAME I/O object → more than 1 DPOR path."""
        from frontrun._dpor import PyDporEngine

        engine = PyDporEngine(num_threads=2, preemption_bound=2, max_branches=100)
        execution = engine.begin_execution()

        engine.schedule(execution)
        engine.report_io_access(execution, 0, 100, "write")

        engine.schedule(execution)
        engine.report_io_access(execution, 1, 100, "write")  # SAME object!

        execution.finish_thread(0)
        engine.schedule(execution)
        execution.finish_thread(1)

        assert engine.next_execution(), "Conflicting I/O objects should need more than 1 path"

    def test_many_independent_io_objects_single_path(self) -> None:
        """Two threads each writing 10 different I/O objects → exactly 1 path."""
        from frontrun._dpor import PyDporEngine

        engine = PyDporEngine(num_threads=2, preemption_bound=2, max_branches=200)
        execution = engine.begin_execution()

        # Thread 0 writes objects 0-9, thread 1 writes objects 10-19
        for i in range(10):
            engine.schedule(execution)
            engine.report_io_access(execution, 0, i, "write")

        for i in range(10, 20):
            engine.schedule(execution)
            engine.report_io_access(execution, 1, i, "write")

        execution.finish_thread(0)
        engine.schedule(execution)
        execution.finish_thread(1)

        assert not engine.next_execution(), "10+10 independent I/O objects should need exactly 1 path"

    def test_redis_resource_ids_independent(self) -> None:
        """Redis resource IDs for different keys map to different DPOR object keys."""
        from frontrun.async_dpor import _make_object_key

        keys_a = [f"ind:{i}" for i in range(10)]
        keys_b = [f"ind:{i}" for i in range(10, 20)]

        resource_ids_a = [f"redis:{k}" for k in keys_a]
        resource_ids_b = [f"redis:{k}" for k in keys_b]

        object_keys_a = {_make_object_key(hash(rid), rid) for rid in resource_ids_a}
        object_keys_b = {_make_object_key(hash(rid), rid) for rid in resource_ids_b}

        assert not (object_keys_a & object_keys_b), "Different Redis keys must produce different DPOR object keys"
