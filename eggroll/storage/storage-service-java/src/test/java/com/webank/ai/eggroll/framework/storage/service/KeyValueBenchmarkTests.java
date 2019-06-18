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
import com.webank.ai.eggroll.core.model.Bytes;
import com.webank.ai.eggroll.framework.storage.service.model.RemoteKeyValueStore;
import com.webank.ai.eggroll.framework.storage.service.model.enums.Stores;
import com.webank.ai.eggroll.framework.storage.service.server.ObjectStoreServicer;
import com.webank.ai.eggroll.framework.storage.service.model.LevelDBStore;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.ServerInterceptors;
import org.junit.*;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.File;
import java.io.IOException;
import java.util.*;

@RunWith(Parameterized.class)
public class KeyValueBenchmarkTests {
    static final int port = 7878;
    static final StoreInfo.StoreInfoBuilder infoBuilder = StoreInfo.builder()
            .nameSpace("testNamespace")
            .tableName("testTable")
            .fragment(0);
    static MockStoreManager storeMgr;
    static File serverDir;
    private KeyValueStore<Bytes, byte[]> store;
    private File dir;

    public KeyValueBenchmarkTests(KeyValueStore store) {
        this.store = store;
    }

    @Parameterized.Parameters
    public static List<Object[]> stores() {
        return Arrays.asList(
                new Object[]{new RemoteKeyValueStore(infoBuilder.type(Stores.IN_MEMORY.name()).build())},
                new Object[]{new RemoteKeyValueStore(infoBuilder.type(Stores.LEVEL_DB.name()).build())},
                new Object[]{Stores.LEVEL_DB.create(infoBuilder.build())},
                new Object[]{Stores.IN_MEMORY.create(infoBuilder.build())}
        );
    }

    @BeforeClass
    public static void startServer() {
        serverDir = TestUtils.tempDirectory();
        storeMgr = new MockStoreManager(serverDir.getAbsolutePath());
        final Server objectStoreServer = ServerBuilder.forPort(port)
                .addService(ServerInterceptors.intercept(new ObjectStoreServicer(storeMgr), new ObjectStoreServicer.KvStoreInterceptor()))
                .maxInboundMessageSize(32 * 1024 * 1024).build();
        MockObjectStoreServer(objectStoreServer);
    }

    @AfterClass
    public static void cleanUp() throws IOException {
        TestUtils.delete(serverDir);
    }

    static void MockObjectStoreServer(Server objectStoreServer) {
        final Thread serverThread = new Thread() {
            @Override
            public void run() {
                try {
                    objectStoreServer.start();
                    objectStoreServer.awaitTermination();
                } catch (IOException | InterruptedException e) {
                    e.printStackTrace();
                }
            }
        };
        serverThread.start();
    }

    @Before
    public void setUp() {
        dir = TestUtils.tempDirectory();
        System.out.println("temporary directory: " + dir.getAbsolutePath());
        Properties properties = new Properties();
        properties.put("host", "localhost");
        properties.put("port", port);
        properties.put(LevelDBStore.DATA_DIR, dir.getAbsolutePath());
        store.init(properties);
    }

    @After
    public void destroy() {
        store.destroy();
        try {
            System.out.println("cleaning directory: " + dir.getAbsolutePath());
            TestUtils.delete(dir);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void benchmark() throws IOException {
        Map<StoreBenchmark.Flag, Object> flags = new EnumMap<>(StoreBenchmark.Flag.class);
        for (StoreBenchmark.Flag flag : StoreBenchmark.Flag.values()) {
            flags.put(flag, flag.getDefaultValue());
        }
        StoreBenchmark benchmark = new StoreBenchmark(flags, store);
        benchmark.run();
    }
}
