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

public abstract class BaseMapPool<K, V> implements MapPool<K, V> {
    public abstract void put(K key, V value);

    public abstract void putIfAbsent(K key, V value);

    public abstract void putAll(Map<K, V> kv);

    public abstract V get(K key);

    @Override
    public void put(K key, V value, boolean onlyIfAbsent) {
        if (onlyIfAbsent) {
            this.putIfAbsent(key, value);
        } else {
            this.put(key, value);
        }
    }
}
