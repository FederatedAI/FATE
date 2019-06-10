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

package com.webank.ai.eggroll.framework.storage.service;

import com.webank.ai.eggroll.core.io.KeyValueStore;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.io.StoreManager;
import com.webank.ai.eggroll.core.model.Bytes;
import com.webank.ai.eggroll.framework.storage.service.model.enums.Stores;
import com.webank.ai.eggroll.framework.storage.service.model.LevelDBStore;

import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

public class MockStoreManager implements StoreManager {
    String dir;

    private Map<StoreInfo, KeyValueStore> storeMap;

    public MockStoreManager(String dir) {
        this.dir = dir;
        this.storeMap = new HashMap<>();
    }


    public void destroy() {
        for (KeyValueStore<Bytes, byte[]> store : storeMap.values()) {
            store.destroy();
        }
        storeMap.clear();
    }

    @Override
    public KeyValueStore<Bytes, byte[]> getStore(StoreInfo info) {
        return storeMap.get(info);
    }

    @Override
    public KeyValueStore<Bytes, byte[]> createIfMissing(StoreInfo info) {
        if (storeMap.containsKey(info)) {
            return storeMap.get(info);
        }
        Stores stores = Stores.valueOf(info.getType());
        KeyValueStore store = stores.create(info);
        Properties properties = new Properties();
        properties.put(LevelDBStore.DATA_DIR, dir);
        store.init(properties);
        storeMap.put(info, store);
        return store;
    }

    @Override
    public void destroy(StoreInfo info) {
        KeyValueStore store = storeMap.remove(info);
        if (store != null)
            store.destroy();
    }
}
