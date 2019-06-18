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

import com.webank.ai.eggroll.core.constant.RuntimeConstants;
import com.webank.ai.eggroll.framework.storage.service.manager.LMDBStoreManager;
import com.webank.ai.eggroll.framework.storage.service.server.LMDBServicer;
import com.webank.ai.eggroll.framework.storage.service.server.ObjectStoreServicer;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.ServerInterceptors;

public class MockRemoteKeyValueStore {
    private static final LMDBStoreManager storeMgr = new LMDBStoreManager(RuntimeConstants.getDefaultDataDir());
    // private static final MockStoreManager storeMgr = new MockStoreManager(RuntimeConstants.getDefaultDataDir());
    private static final int port = 7878;

    public static void main(String[] args) {
        final Server objectStoreServer = ServerBuilder.forPort(port)
                .addService(ServerInterceptors.intercept(new LMDBServicer(storeMgr), new ObjectStoreServicer.KvStoreInterceptor()))
                .maxInboundMessageSize(32 * 1024 * 1024).build();

        KeyValueBenchmarkTests.MockObjectStoreServer(objectStoreServer);
    }
}
