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


import com.webank.ai.eggroll.core.io.*;
import com.webank.ai.eggroll.core.model.Bytes;
import com.webank.ai.eggroll.core.serdes.impl.POJOUtils;
import com.webank.ai.eggroll.framework.storage.service.manager.LMDBStoreManager;
import com.webank.ai.eggroll.framework.storage.service.model.LevelDBStore;
import com.webank.ai.eggroll.framework.storage.service.model.RemoteKeyValueStore;
import com.webank.ai.eggroll.framework.storage.service.model.enums.Stores;
import com.webank.ai.eggroll.framework.storage.service.server.LMDBServicer;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.ServerInterceptors;
import org.junit.*;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.junit.Assert.*;

@RunWith(Parameterized.class)
public class KeyValueStoreTests {
    static final StoreInfo.StoreInfoBuilder infoBuilder = StoreInfo.builder()
            .nameSpace("testNamespace")
            .tableName("testTable1")
            .fragment(0);
    private static final TestUtils.RandomGenerator generator = new TestUtils.RandomGenerator();
    private static final int port = 7878;
    static File serverDir;
    private static StoreManager storeMgr;
    final private KeyValueStore<Bytes, byte[]> store;
    private File dir;

    public KeyValueStoreTests(KeyValueStore<Bytes, byte[]> store) {
        this.store = store;
    }

    @Parameterized.Parameters
    public static List<Object[]> stores() {
        return Arrays.asList(
//                new Object[]{Stores.REDIS.create(infoBuilder.build())},
                new Object[]{new RemoteKeyValueStore(infoBuilder.type(Stores.IN_MEMORY.name()).build())},
                new Object[]{new RemoteKeyValueStore(infoBuilder.type(Stores.LEVEL_DB.name()).build())},
                new Object[]{Stores.LEVEL_DB.create(infoBuilder.build())},
                new Object[]{Stores.IN_MEMORY.create(infoBuilder.build())}
        );
    }

    @BeforeClass
    public static void startServer() {
        serverDir = TestUtils.tempDirectory();
//        storeMgr = new MockStoreManager(serverDir.getAbsolutePath());
        storeMgr = new LMDBStoreManager(serverDir.getAbsolutePath());
        final Server objectStoreServer = ServerBuilder.forPort(port)
//                .addService(ServerInterceptors.intercept(new ObjectStoreServicer(storeMgr), new ObjectStoreServicer.KvStoreInterceptor()))
                .addService(ServerInterceptors.intercept(new LMDBServicer(storeMgr), new LMDBServicer.KvStoreInterceptor()))
                .maxInboundMessageSize(32 * 1024 * 1024).build();

        KeyValueBenchmarkTests.MockObjectStoreServer(objectStoreServer);
    }

    @AfterClass
    public static void cleanUp() throws IOException {
        TestUtils.delete(serverDir);
    }

    @Before
    public void setUp() {
        this.dir = TestUtils.tempDirectory();
        Properties properties = new Properties();
        properties.put("host", "localhost");
        properties.put("port", port);
        properties.put(LevelDBStore.DATA_DIR, dir.getAbsolutePath());
        store.init(properties);
    }

    @After
    public void destroy() {
        store.destroy();
        storeMgr.destroy();
        try {
            TestUtils.delete(dir);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private long printElapsed(long start, String title) {
        long current = System.currentTimeMillis();
        System.out.println(title + " : " + (current - start));
        return current;
    }


    private int countAll() {
        long start = System.currentTimeMillis();
        KeyValueIterator<Bytes, byte[]> iter = store.all();
        int count = 0;
        while (iter.hasNext()) {
            iter.next();
            count++;
        }
        printElapsed(start, "count all");
        return count;
    }

    @Test
    public void rangeSearchCorrectness() {
        char start = '0';
        char end = 'z';
        byte mid = (byte) ((start + end) / 2);
        try {

            for (char alphabet = end; alphabet >= start; alphabet -= 1) {
                byte[] raw = new byte[]{(byte) alphabet};
                store.put(Bytes.wrap(raw), raw);
            }
            Bytes from = Bytes.wrap(new byte[]{(byte) '1'});
            Bytes to = Bytes.wrap(new byte[]{(byte) mid});
            KeyValueIterator<Bytes, byte[]> rangeIter = store.range(from, to);
            int count = 0;
            while (rangeIter.hasNext()) {
                KeyValue<Bytes, byte[]> item = rangeIter.next();
//                System.out.println(item.key);
                count++;
            }
            assertEquals(35, count);
            store.flush();
            countAll();
        } catch (UnsupportedOperationException e) {
            // redis goes here
        }

    }

    @Test
    public void countAllCorrectness() {
        pourNumbers(500);
        assertEquals(500, countAll());
        for (int i = 0; i < 500; i += 5) {
            byte[] rawKey = StoreBenchmark.formatNumber(i);
            Bytes wrappedKey = Bytes.wrap(rawKey);
            store.delete(wrappedKey);
        }

        assertEquals(400, countAll());
    }


    @Test
    public void peekShouldNotAffectNext() {
        long start = System.currentTimeMillis();
        pourNumbers(10000);
        start = printElapsed(start, "pour10000");
        for (int i = 0; i < 10000; i += 5) {
            byte[] rawKey = StoreBenchmark.formatNumber(i);
            Bytes wrappedKey = Bytes.wrap(rawKey);
            store.delete(wrappedKey);
        }
        start = printElapsed(start, "delete 10000");
        Random rand = new Random();
        KeyValueIterator iter = store.all();
        int cnt = 0;
        while (iter.hasNext()) {
            int peekTimes = 0;
            while (peekTimes++ < rand.nextInt(100)) {
                if (peekTimes % 2 == 0) {
                    iter.peekNextKey();
                } else {
                    iter.hasNext();
                }
            }
            iter.next();
            cnt++;
        }
        printElapsed(start, "iter all");
        assertEquals(8000, cnt);
    }

    @Test
    public void putIfAbsentCorrectness() {
        final Bytes keyBytes = new Bytes("keyTest".getBytes());
        final byte[] valueBytes = "A".getBytes();
        final byte[] valueNewBytes = "B".getBytes();
        long start = System.currentTimeMillis();
        byte[] shouldBeNull = store.putIfAbsent(keyBytes, valueBytes);
        start = printElapsed(start, "first put");
        byte[] oldBytes = store.putIfAbsent(keyBytes, valueNewBytes);
        printElapsed(start, "second put");
        assertNull(shouldBeNull);
        assertEquals("A", new String(oldBytes));
    }

    @Test
    public void shouldDelete() {
        final Bytes keyBytes = new Bytes("keyTest".getBytes());
        final byte[] valueBytes = "A".getBytes();

        store.put(keyBytes, valueBytes);
        byte[] oldBytes = store.delete(keyBytes);
        byte[] shouldBeNull = store.get(keyBytes);
        assertNull(shouldBeNull);
        assertEquals("A", new String(oldBytes));
    }

    @Test
    public void shouldThrowNullPointerExceptionOnKeyNullPut() {
        try {
            store.put(null, "someVal".getBytes());
            fail("Should have thrown NullPointerException on null put()");
        } catch (final NullPointerException e) {
            // good
        }
    }

    @Test
    public void shouldThrowNullPointerExceptionOnKeyNullPutAll() {
        List<KeyValue<Bytes, byte[]>> list = new ArrayList<>();
        list.add(POJOUtils.buildKeyValue(Bytes.wrap("a".getBytes()), "a".getBytes()));
        list.add(new KeyValue<>(null, "b".getBytes()));

        try {
            store.putAll(list);
            fail("Should have thrown NullPointerException on null putAll()");
        } catch (final NullPointerException e) {

        }
    }

    @Test
    public void shouldThrowNullPointerExceptionOnNullGet() {
        try {
            store.get(null);
            fail("Should have thrown NullPointerException on null get()");
        } catch (final NullPointerException e) {
            // good
        }
    }

    @Test
    public void shouldThrowNullPointerExceptionOnNullDelete() {
        try {
            store.delete(null);
            fail("Should have thrown NullPointerException on null delete()");
        } catch (final NullPointerException e) {
            // good
        }
    }

//    @Test
//    public void multiThread() throws InterruptedException {
//        Thread t1 = new Thread() {
//            @Override
//            public void run() {
//                for (int i = 0; i < 100000; i++) {
//                    byte[] bytes = (i + "").getBytes();
//                    store.put(Bytes.wrap(bytes), bytes);
//                }
//            }
//        };
//        Thread t2 = new Thread() {
//            @Override
//            public void run() {
//                for (int i = 0; i < 100000; i++) {
//                    byte[] bytes = (i + "").getBytes();
//                    store.delete(Bytes.wrap(bytes));
//                }
//            }
//        };
//        Thread t3 = new Thread() {
//            @Override
//            public void run() {
//                for (int i = 0; i < 100000; i++) {
//                    byte[] bytes = (i + "").getBytes();
//                    store.get(Bytes.wrap(bytes));
//                }
//            }
//        };
//        Thread t4 = new Thread() {
//            @Override
//            public void run() {
//                for (int i = 0; i < 100000; i++) {
//                    byte[] bytes = (i + "").getBytes();
//                    store.get(Bytes.wrap(bytes));
//                }
//            }
//        };
//        t1.start();
//        t2.start();
//        t3.start();
//        t4.start();
//        t1.join();
//        t2.join();
//        t3.join();
//        t4.join();
//    }

    private void pourNumbers(int count) {
        List<KeyValue<Bytes, byte[]>> toPut = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            byte[] rawKey = StoreBenchmark.formatNumber(i);
            Bytes wrappedKey = Bytes.wrap(rawKey);
            toPut.add(POJOUtils.buildKeyValue(wrappedKey, generator.generate(1024)));
        }
        store.putAll(toPut);
    }

}
