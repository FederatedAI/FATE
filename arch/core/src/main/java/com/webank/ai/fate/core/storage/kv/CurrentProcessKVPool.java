/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.webank.ai.fate.core.storage.kv;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public abstract class CurrentProcessKVPool<K, V> extends BaseKVPool<K, V> {
    protected ConcurrentHashMap<K, V> dataPool;

    public CurrentProcessKVPool(){
        this.dataPool = new ConcurrentHashMap<>();
    }

    @Override
    public void put(K key, V value){
        this.dataPool.put(key, value);
    }

    @Override
    public void putIfAbsent(K key, V value){
        this.dataPool.putIfAbsent(key, value);
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
