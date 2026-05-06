import time

from hubspot_agent.cache import SchemaCache


def test_cache_get_set(tmp_path):
    cache = SchemaCache("123", base_dir=tmp_path)
    cache.set("objects", {"contacts": ["email"]})
    assert cache.get("objects") == {"contacts": ["email"]}


def test_cache_miss(tmp_path):
    cache = SchemaCache("123", base_dir=tmp_path)
    assert cache.get("nonexistent") is None


def test_cache_ttl_expiration(tmp_path, monkeypatch):
    cache = SchemaCache("123", base_dir=tmp_path)
    cache.set("objects", {"contacts": ["email"]})
    fixed_time = time.time() + 4000
    monkeypatch.setattr(time, "time", lambda: fixed_time)
    assert cache.get("objects") is None


def test_cache_invalidate(tmp_path):
    cache = SchemaCache("123", base_dir=tmp_path)
    cache.set("objects", {"contacts": ["email"]})
    cache.invalidate("objects")
    assert cache.get("objects") is None


def test_cache_refresh_all(tmp_path):
    cache = SchemaCache("123", base_dir=tmp_path)
    cache.set("objects", {"contacts": ["email"]})
    cache.refresh_all()
    assert cache.get("objects") is None


def test_cache_refresh_domain(tmp_path):
    cache = SchemaCache("123", base_dir=tmp_path)
    cache.set("objects", {"contacts": ["email"]})
    cache.set("pipelines", {"stages": []})
    cache.refresh_domain("objects")
    assert cache.get("objects") is None
    assert cache.get("pipelines") == {"stages": []}
