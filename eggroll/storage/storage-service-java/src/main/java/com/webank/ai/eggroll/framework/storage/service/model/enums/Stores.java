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

package com.webank.ai.eggroll.framework.storage.service.model.enums;

import com.webank.ai.eggroll.core.io.KeyValueBytesStoreSupplier;
import com.webank.ai.eggroll.core.io.KeyValueStore;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.framework.storage.service.model.InMemoryBytesStoreSupplier;
import com.webank.ai.eggroll.framework.storage.service.model.LMDBSupplier;
import com.webank.ai.eggroll.framework.storage.service.model.LevelDbKeyValueBytesStoreSupplier;

public enum Stores {
    LEVEL_DB(new LevelDbKeyValueBytesStoreSupplier()),
    IN_MEMORY(new InMemoryBytesStoreSupplier()),
    LMDB(new LMDBSupplier());

    final KeyValueBytesStoreSupplier supplier;

    Stores(KeyValueBytesStoreSupplier supplier) {
        this.supplier = supplier;
    }

    public KeyValueStore create(StoreInfo info) {
        return supplier.get(info);
    }

}
