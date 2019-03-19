package com.webank.ai.fate.core.storage.kv;

import java.util.Map;

public abstract class BaseKVPool<K, V> implements KVPool<K, V> {
    public abstract V put(K key, V value);
    public abstract V putIfAbsent(K key, V value);
    public abstract void putAll(Map<K, V> kv);
    public abstract V get(K key);

    @Override
    public V put(K key, V value, boolean onlyIfAbsent){
        if(onlyIfAbsent){
            return this.putIfAbsent(key, value);
        }
        else{
            return this.put(key, value);
        }
    }
}
