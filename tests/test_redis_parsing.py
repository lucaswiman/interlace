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

    def test_rename_reads_source_writes_dest(self) -> None:
        result = parse_redis_access("RENAME", ("old", "new"))
        assert result.read_keys == ["old"]
        assert result.write_keys == ["new"]


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

    def test_rpoplpush_reads_source_writes_dest(self) -> None:
        result = parse_redis_access("RPOPLPUSH", ("source", "dest"))
        assert result.read_keys == ["source"]
        assert result.write_keys == ["dest"]

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

    def test_eval_with_keys(self) -> None:
        result = parse_redis_access("EVAL", ("script", 2, "key1", "key2", "arg1"))
        assert result.read_keys == ["key1", "key2"]
        assert result.write_keys == ["key1", "key2"]

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
