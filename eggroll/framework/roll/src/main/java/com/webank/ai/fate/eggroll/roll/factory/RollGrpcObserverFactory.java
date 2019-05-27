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

package com.webank.ai.fate.eggroll.roll.factory;

import com.webank.ai.fate.api.eggroll.storage.Kv;
import com.webank.ai.fate.core.io.StoreInfo;
import com.webank.ai.fate.eggroll.meta.service.dao.generated.model.Node;
import com.webank.ai.fate.eggroll.roll.api.grpc.observer.kv.roll.RollKvPutAllServerRequestStreamObserver;
import com.webank.ai.fate.eggroll.roll.api.grpc.observer.kv.storage.StorageKvPutAllServerRequestStreamObserver;
import io.grpc.stub.StreamObserver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicBoolean;


@Component
@Scope("prototype")
public class RollGrpcObserverFactory {
    @Autowired
    public ApplicationContext applicationContext;

    public RollKvPutAllServerRequestStreamObserver createRollKvPutAllServerRequestStreamObserver(final StreamObserver<Kv.Empty> clientResponseObserver,
                                                                                                 final StoreInfo storeInfo,
                                                                                                 final AtomicBoolean wasReady) {
        return applicationContext.getBean(RollKvPutAllServerRequestStreamObserver.class, clientResponseObserver, storeInfo, wasReady);
    }

    public StorageKvPutAllServerRequestStreamObserver createStoragePutAllRequestStreamObserver(final StreamObserver<Kv.Empty> clientResponseObserver,
                                                                                               final StoreInfo storeInfo,
                                                                                               final Node node) {
        return applicationContext.getBean(StorageKvPutAllServerRequestStreamObserver.class, clientResponseObserver, storeInfo, node);
    }
}
