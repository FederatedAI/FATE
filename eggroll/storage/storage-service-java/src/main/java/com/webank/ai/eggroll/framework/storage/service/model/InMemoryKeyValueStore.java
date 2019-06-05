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

package com.webank.ai.eggroll.framework.storage.service.model;


import com.webank.ai.eggroll.core.error.exception.InvalidStateStoreException;
import com.webank.ai.eggroll.core.io.KeyValue;
import com.webank.ai.eggroll.core.io.KeyValueIterator;
import com.webank.ai.eggroll.core.io.KeyValueStore;
import com.webank.ai.eggroll.core.io.StoreInfo;
import io.grpc.stub.StreamObserver;

import java.util.*;

public class InMemoryKeyValueStore<K, V> implements KeyValueStore<K, V> {
    private final String name;
    private final NavigableMap<K, V> map;
    private volatile boolean open = false;

    public InMemoryKeyValueStore(StoreInfo info) {
        this.name = info.getTableName();
        this.map = new TreeMap<>();
    }

    @Override
    public synchronized void put(K key, V value) {
        if (value == null) {
            this.map.remove(key);
        } else {
            this.map.put(key, value);
        }
    }

    @Override
    public synchronized V putIfAbsent(K key, V value) {
        final V originalValue = get(key);
        if (originalValue == null) {
            put(key, value);
        }
        return originalValue;
    }

    @Override
    public synchronized void putAll(List<KeyValue<K, V>> entries) {
        for (final KeyValue<K, V> entry : entries) {
            put(entry.key, entry.value);
        }
    }

    @Override
    public StreamObserver<KeyValue<K, V>> putAll() {
        return new StreamObserver<KeyValue<K, V>>() {
            @Override
            public void onNext(final KeyValue<K, V> keyValue) {
                put(keyValue.key, keyValue.value);
            }

            @Override
            public void onError(Throwable throwable) {
                // TODO ignore now
            }

            @Override
            public void onCompleted() {

            }
        };
    }

    @Override
    public synchronized V delete(K key) {
        return this.map.remove(key);
    }

    @Override
    public synchronized V get(K key) {
        return this.map.get(key);
    }

    @Override
    public synchronized KeyValueIterator<K, V> range(K from, K to) {
        if (from != null && to != null)
            return new InMemoryKeyValueIterator<>(this.map.subMap(from, false, to, false).entrySet().iterator(), name);
        if (from != null)
            return new InMemoryKeyValueIterator<>(this.map.tailMap(from, false).entrySet().iterator(), name);
        if (to != null)
            return new InMemoryKeyValueIterator<>(this.map.headMap(to, false).entrySet().iterator(), name);
        return all();
    }

    @Override
    public synchronized KeyValueIterator<K, V> all() {
        final TreeMap<K, V> copy = new TreeMap<>(this.map);
        return new InMemoryKeyValueIterator<>(copy.entrySet().iterator(), name);
    }

    @Override
    public synchronized void destroy() {
        close();
    }

    @Override
    public long count() {
        throw new UnsupportedOperationException("count operation not supported");
    }

    @Override
    public synchronized void close() {
        this.map.clear();
        this.open = false;
    }

    @Override
    public boolean persistent() {
        return false;
    }

    @Override
    public void flush() {

    }

    @Override
    public String name() {
        return this.name;
    }

    @Override
    public synchronized void init(Properties properties) {
        this.open = true;
    }


    @Override
    public boolean isOpen() {
        return this.open;
    }

    private static class InMemoryKeyValueIterator<K, V> implements KeyValueIterator<K, V> {
        private final Iterator<Map.Entry<K, V>> iter;
        private final String name;
        private KeyValue<K, V> next;
        private boolean open = true;

        private InMemoryKeyValueIterator(final Iterator<Map.Entry<K, V>> iter, final String name) {
            this.iter = iter;
            this.name = name;
        }

        @Override
        public boolean hasNext() {
            if (!open) {
                throw new InvalidStateStoreException(String.format("Store %s has closed", name));
            }
            if (next != null) {
                return true;
            }

            if (!iter.hasNext()) {
                return false;
            }

            Map.Entry<K, V> entry = iter.next();
            next = new KeyValue<>(entry.getKey(), entry.getValue());
            return true;
        }

        @Override
        public KeyValue<K, V> next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            final KeyValue<K, V> result = next;
            next = null;
            return result;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("remove() is not supported in " + getClass().getName());
        }

        @Override
        public void close() {
            open = false;
        }

        @Override
        public K peekNextKey() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return next.key;
        }
    }
}
