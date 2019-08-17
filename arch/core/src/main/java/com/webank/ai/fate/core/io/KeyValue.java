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

import java.util.Objects;

public class KeyValue<K, V> {

    public final K key;
    public final V value;


    public KeyValue(final K key, final V value) {
        this.key = key;
        this.value = value;
    }


    public static <K, V> KeyValue<K, V> pair(final K key, final V value) {
        return new KeyValue<>(key, value);
    }

    @Override
    public String toString() {
        return "KeyValue(" + key + ", " + value + ")";
    }

    @Override
    public boolean equals(final Object obj) {
        if (this == obj)
            return true;

        if (!(obj instanceof KeyValue)) {
            return false;
        }

        final KeyValue other = (KeyValue) obj;
        return (key == null ? other.key == null : key.equals(other.key))
                && (value == null ? other.value == null : value.equals(other.value));
    }

    @Override
    public int hashCode() {
        return Objects.hash(key, value);
    }

}