package com.webank.ai.fate.core.storage.kv;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public abstract class CurrentProcessKVPool<K, V> extends BaseKVPool<K, V> {
    protected ConcurrentHashMap<K, V> dataPool;

    public CurrentProcessKVPool(){
        this.dataPool = new ConcurrentHashMap<>();
    }

    @Override
    public V put(K key, V value){
        return this.dataPool.put(key, value);
    }

    @Override
    public V putIfAbsent(K key, V value){
        return this.dataPool.putIfAbsent(key, value);
    }

    @Override
    public void putAll(Map<K, V> kv){
        this.dataPool.putAll(kv);
    }

    @Override
    public V get(K key){
        return this.dataPool.get(key);
    }
}
