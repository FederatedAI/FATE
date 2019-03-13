package com.webank.ai.fate.common.storage.kv;

import java.util.Map;

public interface KVPool<K, V>{
    V put(K key, V value);
    V put(K key, V value, boolean onlyIfAbsent);

    V putIfAbsent(K key, V value);

    void putAll(Map<K, V> kv);

    V get(K key);
}
