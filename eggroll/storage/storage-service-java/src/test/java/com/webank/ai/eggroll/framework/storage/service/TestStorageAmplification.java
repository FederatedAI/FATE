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
import com.webank.ai.eggroll.core.io.KeyValue;
import com.webank.ai.eggroll.core.io.KeyValueIterator;
import com.webank.ai.eggroll.core.io.KeyValueStore;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.model.Bytes;
import com.webank.ai.eggroll.framework.storage.service.model.RemoteKeyValueStore;
import com.webank.ai.eggroll.framework.storage.service.model.enums.Stores;
import org.apache.commons.lang3.RandomStringUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.util.Properties;

public class TestStorageAmplification {
    private static final TestUtils.RandomGenerator generator = new TestUtils.RandomGenerator();
    private static final String dirPath = RuntimeConstants.getDefaultDataDir();
    private static final MockStoreManager storeMgr = new MockStoreManager(dirPath);
    private static final Logger LOGGER = LogManager.getLogger();
    final private KeyValueStore<Bytes, byte[]> store;
    private Stores type;
    private File dir;
    private int port = 7778;
    private String namespace = "storage_test_namespace";
    private String name = "storage_test_100m";
    private String clusterCommTableName = "__clustercomm__";
    private String jobid1 = "jobid1";

    public TestStorageAmplification() {
        this.store = new RemoteKeyValueStore(StoreInfo.builder()
                .type(Stores.LMDB.name())
                //.nameSpace(clusterCommTableName)
                .nameSpace(namespace)
                //.nameSpace(jobid1)
                .tableName(name)
                //.tableName(name + "_recv")
                //.tableName(clusterCommTableName)
                .fragment(0)
                .build());
        //this.type = Stores.LMDB;
    }

    @Before
    public void setUp() {
        this.dir = TestUtils.tempDirectory();

        Properties properties = new Properties();
        properties.put("host", "localhost");
        properties.put("port", port);
        store.init(properties);
        // storeMgr.init(properties);
    }

    @Test
    public void putOnce() {
        String str = RandomStringUtils.randomAlphanumeric(100 << 20);
        byte[] strByte = str.getBytes();

        System.out.println(str.length());
        System.out.println(strByte.length);

        store.put(Bytes.wrap("100m".getBytes()), strByte);
    }

    @Test
    public void put1MChunk() {
        for (int i = 0; i < 100; ++i) {
            //String str = RandomStringUtils.randomAlphanumeric(1 << 20);
            String str = StringUtils.repeat("0", 1 << 20);
            byte[] strByte = str.getBytes();

            System.out.println(i);

            String key = "key_" + i;
            store.put(Bytes.wrap(key.getBytes()), strByte);
        }
    }

    @Test
    public void putJobObject() {
        store.put(Bytes.wrapUtf8String("30-name"), RandomStringUtils.random(16384).getBytes());
    }

    @Test
    public void get() {
        byte[] result = store.get(Bytes.wrap("hello".getBytes()));

        LOGGER.info("result: {}", new String(result));
    }

    @Test
    public void all() {
        KeyValueIterator<Bytes, byte[]> iterator = store.all();
        while (iterator.hasNext()) {
            KeyValue<Bytes, byte[]> kv = iterator.next();
            LOGGER.info("key: {}, value: {}", new String(kv.key.get()), new String(kv.value));
        }
    }

    @Test
    public void count() {
        long result = store.count();

        LOGGER.info("count: {}", result);
    }

    @Test
    public void destroy() {
        store.destroy();
    }
}
