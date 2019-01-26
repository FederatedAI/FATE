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

package com.webank.ai.fate.core.io;

import io.grpc.stub.StreamObserver;

import java.util.List;


// todo: reduce dependencies
public interface KeyValueStore<K, V> extends StateStore {

    void put(K key, V value);

    V putIfAbsent(K key, V value);

    void putAll(List<KeyValue<K, V>> entries);

    StreamObserver<KeyValue<K, V>> putAll();

    V delete(K key);

    V get(K key);

    // open interval
    KeyValueIterator<K, V> range(K from, K to);

    KeyValueIterator<K, V> all();

    void destroy();

    long count();
}
