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

package com.webank.ai.fate.eggroll.roll.service.async.storage;

import com.webank.ai.fate.core.io.StoreInfo;
import com.webank.ai.fate.eggroll.meta.service.dao.generated.model.Node;
import com.webank.ai.fate.eggroll.roll.api.grpc.client.StorageServiceClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.Callable;

@Component
@Scope("prototype")
public abstract class BaseStorageProcessor<R, V> implements Callable<V> {
    @Autowired
    protected StorageServiceClient storageServiceClient;

    protected R request;
    protected StoreInfo storeInfo;
    protected Node node;

    public BaseStorageProcessor(R request, StoreInfo storeInfo, Node node) {
        this.request = request;
        this.storeInfo = storeInfo;
        this.node = node;
    }
}
