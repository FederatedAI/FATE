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

package com.webank.ai.eggroll.core.error.exception;

import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.core.io.StoreInfo;

public class StorageNotExistsException extends RuntimeException {

    public StorageNotExistsException(StoreInfo storeInfo) {
        super(buildWithStoreInfo(storeInfo));
    }

    public StorageNotExistsException(StorageBasic.StorageLocator storageLocator) {
        this(StoreInfo.fromStorageLocator(storageLocator));
    }

    private static String buildWithStoreInfo(StoreInfo storeInfo) {
        StringBuilder sb = new StringBuilder();
        sb.append("Store with namespace: ")
                .append(storeInfo.getNameSpace())
                .append(", name: ")
                .append(storeInfo.getTableName())
                .append(", fragment: ")
                .append(storeInfo.getFragment())
                .append(", type: ")
                .append(storeInfo.getType())
                .append(" does not exist");

        return sb.toString();
    }
}
