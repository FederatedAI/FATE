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

package com.webank.ai.fate.eggroll.roll.api.grpc.client;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.eggroll.storage.Kv;
import com.webank.ai.fate.api.eggroll.storage.StorageBasic;
import com.webank.ai.fate.core.factory.GrpcServerFactory;
import com.webank.ai.fate.core.io.StoreInfo;
import com.webank.ai.fate.core.model.Bytes;
import com.webank.ai.fate.core.server.ServerConf;
import com.webank.ai.fate.core.utils.ToStringUtils;
import com.webank.ai.fate.eggroll.roll.service.model.OperandBroker;
import com.webank.ai.fate.eggroll.storage.service.model.enums.Stores;
import org.apache.commons.lang3.RandomStringUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath*:applicationContext-roll.xml"})
public class TestRollKvServiceClient {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private RollKvServiceClient rollKvServiceClient;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private GrpcServerFactory grpcServerFactory;
    @Autowired
    private ServerConf serverConf;


    private String name = "api_create_name";
    private String namespace = "api_create_namespace";
    private String jobid1 = "jobid1";
    private String federationTable = "__federation__";
    private StoreInfo storeInfo;

    @Before
    public void init() throws Exception {
        serverConf = grpcServerFactory.parseConfFile("/Users/max-webank/git/floss/project/eggroll/roll/src/main/resources/roll.properties");
        storeInfo = StoreInfo.builder()
                .type(Stores.LMDB.name())
                .nameSpace(namespace)
                .tableName(name)
                .build();
    }

    @Test
    public void testCreate() {
        Kv.CreateTableInfo.Builder createTableInfoBuilder = Kv.CreateTableInfo.newBuilder();
        StorageBasic.StorageLocator.Builder storageLocatorBuilder = StorageBasic.StorageLocator.newBuilder();
        storageLocatorBuilder.setType(StorageBasic.StorageType.LMDB)
                .setNamespace(namespace)
                .setName(name);

        createTableInfoBuilder.setStorageLocator(storageLocatorBuilder.build())
                .setFragmentCount(10);

        rollKvServiceClient.create(createTableInfoBuilder.build());
    }

    @Test
    public void testPut() {
        Kv.Operand.Builder operandBuilder = Kv.Operand.newBuilder();

        Kv.Operand operand = operandBuilder.setKey(ByteString.copyFromUtf8("lemon")).setValue(ByteString.copyFromUtf8("tea")).build();

        StoreInfo storeInfo = StoreInfo.builder()
                .type(Stores.LMDB.name())
                .nameSpace(namespace)
                .tableName(name)
                .build();

        rollKvServiceClient.put(operand, storeInfo);
    }

    @Test
    public void testPutIfAbsent() {
        Kv.Operand.Builder operandBuilder = Kv.Operand.newBuilder();

        Kv.Operand operand = operandBuilder.setKey(ByteString.copyFromUtf8("time")).setValue(ByteString.copyFromUtf8(String.valueOf(System.currentTimeMillis()))).build();

        StoreInfo storeInfo = StoreInfo.builder()
                .type(Stores.LMDB.name())
                .nameSpace(namespace)
                .tableName(name)
                .build();

        Kv.Operand result = rollKvServiceClient.putIfAbsent(operand, storeInfo);

        LOGGER.info("putIfAbsent result: {}", result);
    }

    @Test
    public void testPutAll() {


        OperandBroker operandBroker = new OperandBroker();
        Kv.Operand.Builder operandBuilder = Kv.Operand.newBuilder();
        Kv.Operand operand1 = operandBuilder.setKey(ByteString.copyFromUtf8("time")).setValue(ByteString.copyFromUtf8(String.valueOf(System.currentTimeMillis()))).build();
        Kv.Operand operand2 = operandBuilder.setKey(ByteString.copyFromUtf8("key")).setValue(ByteString.copyFromUtf8("value")).build();
        Kv.Operand operand3 = operandBuilder.setKey(ByteString.copyFromUtf8("hello")).setValue(ByteString.copyFromUtf8("world")).build();
        Kv.Operand operand4 = operandBuilder.setKey(ByteString.copyFromUtf8("iter")).setValue(ByteString.copyFromUtf8("ator")).build();
        Kv.Operand operand5 = operandBuilder.setKey(ByteString.copyFromUtf8("happy")).setValue(ByteString.copyFromUtf8("holidays")).build();

        StoreInfo storeInfo = StoreInfo.builder()
                .type(Stores.LMDB.name())
                .nameSpace(namespace)
                .tableName(name)
                .build();

        operandBroker.put(operand1);
        operandBroker.put(operand2);
        operandBroker.put(operand3);
        operandBroker.put(operand4);
        operandBroker.put(operand5);

        operandBroker.setFinished();
        rollKvServiceClient.putAll(operandBroker, storeInfo);
    }

    @Test
    public void testPutAllToSameDb() {
        OperandBroker operandBroker = new OperandBroker();
        Kv.Operand.Builder operandBuilder = Kv.Operand.newBuilder();

        StoreInfo storeInfo = StoreInfo.builder()
                .type(Stores.LMDB.name())
                .nameSpace(namespace)
                .tableName(name)
                .build();

        Kv.Operand operand = null;
        for (int i = 0; i < 100; ++i) {
            //operand = operandBuilder.setKey(ByteString.copyFromUtf8(RandomStringUtils.randomAlphanumeric(20))).setValue(ByteString.copyFromUtf8("v" + i)).build();
            operand = operandBuilder.setKey(ByteString.copyFromUtf8("k" + i)).setValue(ByteString.copyFromUtf8("v" + i)).build();
            operandBroker.put(operand);
        }

        operandBroker.setFinished();
        rollKvServiceClient.putAll(operandBroker, storeInfo);

        operandBroker = new OperandBroker();
        for (int i = 1000; i < 1100; ++i) {
            operand = operandBuilder.setKey(ByteString.copyFromUtf8("k" + i)).setValue(ByteString.copyFromUtf8("v" + i)).build();
            operandBroker.put(operand);
        }

        operandBroker.setFinished();
        rollKvServiceClient.putAll(operandBroker, storeInfo);
    }

    @Test
    public void testPutAllMany() throws Exception {
        OperandBroker operandBroker = new OperandBroker();
        Kv.Operand.Builder operandBuilder = Kv.Operand.newBuilder();

        StoreInfo storeInfo = StoreInfo.builder()
                .type(Stores.LMDB.name())
                .nameSpace(namespace)
                .tableName(name)
                .build();


        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                rollKvServiceClient.putAll(operandBroker, storeInfo);
            }
        });

        thread.start();

        Kv.Operand operand = null;
        int resetInterval = 100000;
        int curCount = 0;
        for (int i = 0; i < 100; ++i) {
            if (curCount <= 0) {
                curCount = resetInterval;
                LOGGER.info("current: {}", i);
            }

            --curCount;
            //operand = operandBuilder.setKey(ByteString.copyFromUtf8(RandomStringUtils.randomAlphanumeric(20))).setValue(ByteString.copyFromUtf8("v" + i)).build();
            operand = operandBuilder.setKey(ByteString.copyFromUtf8("k" + i)).setValue(ByteString.copyFromUtf8("v" + i)).build();
            operandBroker.put(operand);
        }

        operandBroker.setFinished();

        thread.join();
        //rollKvServiceClient.putAll(operandBroker, storeInfo);
    }

    @Test
    public void testDelete() {
        Kv.Operand.Builder operandBuilder = Kv.Operand.newBuilder();

        Kv.Operand operand = operandBuilder.setKey(ByteString.copyFromUtf8("time")).setValue(ByteString.copyFromUtf8(String.valueOf(System.currentTimeMillis()))).build();

        StoreInfo storeInfo = StoreInfo.builder()
                .type(Stores.LMDB.name())
                .nameSpace(namespace)
                .tableName(name)
                .build();

        rollKvServiceClient.put(operand, storeInfo);
        Kv.Operand result = rollKvServiceClient.delete(operand, storeInfo);

        LOGGER.info("delete result: {}", result);
    }

    @Test
    public void testGet() {
        Kv.Operand.Builder operandBuilder = Kv.Operand.newBuilder();

        Kv.Operand operand = operandBuilder.setKey(ByteString.copyFromUtf8("happy")).setValue(ByteString.copyFromUtf8(String.valueOf(System.currentTimeMillis()))).build();

        StoreInfo storeInfo = StoreInfo.builder()
                .type(Stores.LMDB.name())
                .nameSpace(namespace)
                .tableName(name)
                .build();

        Kv.Operand result = rollKvServiceClient.get(operand, storeInfo);

        LOGGER.info("get result: {}", result);
    }

    @Test
    public void testIterate() throws Exception {
        Kv.Range range = Kv.Range.newBuilder().setStart(ByteString.copyFromUtf8("")).setEnd(ByteString.copyFromUtf8("")).setMinChunkSize(1000).build();
        StoreInfo storeInfo = StoreInfo.builder()
                .type(Stores.LMDB.name())
                .nameSpace(namespace)
                .tableName(name)
                .build();
        OperandBroker operandBroker = rollKvServiceClient.iterate(range, storeInfo);
        operandBroker = rollKvServiceClient.iterate(range, storeInfo);

        List<Kv.Operand> operands = Lists.newLinkedList();

        int count = 0;
        Kv.Operand previous = null;
        Kv.Operand current = null;
        int correctCount = 0;
        while (!operandBroker.isClosable()) {
            operandBroker.awaitLatch(1, TimeUnit.SECONDS);
            operandBroker.drainTo(operands);
            count += operands.size();

            for (Kv.Operand operand : operands) {
                previous = current;
                current = operand;

                if (previous != null && current != null) {
                    Bytes previousBytes = Bytes.wrap(previous.getKey());
                    Bytes currentBytes = Bytes.wrap(current.getKey());

                    if (previousBytes.compareTo(currentBytes) < 0) {
                        ++correctCount;
                    }
                }
                // LOGGER.info("operand: {}", toStringUtils.toOneLineString(operand));
                // LOGGER.info("key: {}, value: {}", operand.getKey().toStringUtf8(), operand.getValue().toStringUtf8());
            }
            operands.clear();
        }

        LOGGER.info("iterate count: {}", count);
        LOGGER.info("correct count: {}", correctCount);
    }

    @Test
    public void testIterateSegment() throws Exception {
        Kv.Range range = Kv.Range.newBuilder().setStart(ByteString.copyFromUtf8("")).setEnd(ByteString.copyFromUtf8("")).setMinChunkSize(2<<20).build();
        StoreInfo storeInfo = StoreInfo.builder()
                .type(Stores.LMDB.name())
                .nameSpace(namespace)
                .tableName(name)
                .build();

        Set<String> bsSet = Sets.newConcurrentHashSet();
        int batchCount = 0;
        int count = 0;
        int correctCount = 0;
        Kv.Operand previous = null;
        Kv.Operand current = null;
        LinkedList<Kv.Operand> operands = Lists.newLinkedList();

        boolean hasProcessed = false;
        while (!hasProcessed || !operands.isEmpty()) {
            hasProcessed = true;
            operands.clear();
            LOGGER.info("range start: {}, end: {}, count: {}, correct count: {}, set count: {}",
                    range.getStart().toStringUtf8(), range.getEnd().toStringUtf8(), count, correctCount, bsSet.size());
            OperandBroker operandBroker = rollKvServiceClient.iterate(range, storeInfo);

            while (!operandBroker.isClosable()) {
                operandBroker.awaitLatch(1, TimeUnit.SECONDS);
                operandBroker.drainTo(operands);
                count += operands.size();

                for (Kv.Operand operand : operands) {
                    previous = current;
                    current = operand;
                    bsSet.add(operand.getKey().toStringUtf8());
                    if (previous != null && current != null) {
                        Bytes previousBytes = Bytes.wrap(previous.getKey());
                        Bytes currentBytes = Bytes.wrap(current.getKey());

                        if (previousBytes.compareTo(currentBytes) < 0) {
                            ++correctCount;
                        }
                    }
                    // LOGGER.info("operand: {}", toStringUtils.toOneLineString(operand));
                    // LOGGER.info("key: {}, value: {}", operand.getKey().toStringUtf8(), operand.getValue().toStringUtf8());

                }
                Kv.Operand last = operands.getLast();
                range = range.toBuilder().setStart(last.getKey()).build();

            }
            LOGGER.info("iterate count: {}, set size: {}, correct count: {}", count, bsSet.size(), correctCount);
        }
    }

    @Test
    public void testDestroy() {
        StoreInfo storeInfo = StoreInfo.builder()
                .type(Stores.LMDB.name())
                .nameSpace(namespace)
                .tableName(name)
                .build();

        rollKvServiceClient.destroy(storeInfo);

        LOGGER.info("done destroy");
    }

    @Test
    public void testDestroyAll() {
        StoreInfo storeInfo = StoreInfo.builder()
                .type(Stores.LMDB.name())
                .nameSpace("ce46817e-13bf-11e9-8d62-4a00003fc630")
                .tableName("*")
                .build();

        rollKvServiceClient.destroyAll(storeInfo);

        LOGGER.info("done destroyAll");
    }

    @Test
    public void testCount() {
        StoreInfo storeInfo = StoreInfo.builder()
                .type(Stores.LMDB.name())
                .nameSpace(namespace)
                .tableName(name)
                .build();
        Kv.Count result = rollKvServiceClient.count(storeInfo);
        LOGGER.info("count result: {}", result.getValue());
    }

    @Test
    public void testBigData() {
        ByteString key = ByteString.copyFromUtf8("1M");
        Kv.Operand.Builder operandBuilder = Kv.Operand.newBuilder();
        operandBuilder.setKey(key)
                .setValue(ByteString.copyFromUtf8(StringUtils.repeat("1", 10000000)));

        rollKvServiceClient.put(operandBuilder.build(), storeInfo);

        LOGGER.info("put finished");
        Kv.Operand result = rollKvServiceClient.get(operandBuilder.clear().setKey(key).build(), storeInfo);
        LOGGER.info("get done. length: {}", result.getValue().size());
    }
}
