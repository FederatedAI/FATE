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

package com.webank.ai.eggroll.framework.storage.service.manager;

import com.webank.ai.eggroll.core.io.KeyValueStore;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.io.StoreManager;
import com.webank.ai.eggroll.core.model.Bytes;
import com.webank.ai.eggroll.framework.storage.service.model.LevelDBStore;
import com.webank.ai.eggroll.framework.storage.service.model.enums.Stores;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

@Component
@Scope("prototype")
public class LocalStoreManager implements StoreManager<Bytes, byte[]> {
    private final Map<StoreInfo, KeyValueStore<Bytes, byte[]>> storeMap;
    // for mock
    private final String parentDir;

    public LocalStoreManager() {
        this.storeMap = new ConcurrentHashMap<>();
        this.parentDir = System.getProperty("user.dir");
    }

    public LocalStoreManager(String parentDir) {
        this.storeMap = new ConcurrentHashMap<>();
        this.parentDir = parentDir;
    }

    @Override
    public KeyValueStore<Bytes, byte[]> getStore(StoreInfo info) {
        Objects.requireNonNull(info, "storeInfo should not be null");
        return storeMap.get(info);
    }

    @Override
    public KeyValueStore<Bytes, byte[]> createIfMissing(StoreInfo info) {
        Objects.requireNonNull(info, "storeInfo should not be null");
        if (storeMap.containsKey(info)) {
            return storeMap.get(info);
        }
        Stores type = Stores.valueOf(info.getType());
        synchronized (type) {
            if (!storeMap.containsKey(info)) {
                KeyValueStore keyValueStore = type.create(info);
                Properties properties = new Properties();
                properties.put(LevelDBStore.DATA_DIR, parentDir);
                keyValueStore.init(properties);
                storeMap.putIfAbsent(info, keyValueStore);
            }
        }
        return storeMap.get(info);
    }

    @Override
    public void destroy(StoreInfo info) {
        if (!storeMap.containsKey(info)) {
            return;
        }
        Stores type = Stores.valueOf(info.getType());
        synchronized (type) {
            if (storeMap.containsKey(info)) {
                KeyValueStore store = storeMap.remove(info);
                store.destroy();
            }
        }
    }

    @Override
    public void destroy() {
        for (KeyValueStore<Bytes, byte[]> store : storeMap.values()) {
            store.destroy();
        }
        storeMap.clear();
    }


}
