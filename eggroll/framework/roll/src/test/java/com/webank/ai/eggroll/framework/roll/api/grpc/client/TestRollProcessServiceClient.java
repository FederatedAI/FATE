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

package com.webank.ai.eggroll.framework.roll.api.grpc.client;

import com.google.protobuf.Message;
import com.webank.ai.eggroll.api.computing.processor.Processor;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.core.model.DelayedResult;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.framework.roll.service.model.OperandBroker;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.util.concurrent.TimeUnit;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath*:applicationContext-roll.xml"})
public class TestRollProcessServiceClient {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private RollProcessServiceClient rollProcessServiceClient;
    @Autowired
    private ToStringUtils toStringUtils;
    private Processor.UnaryProcess.Builder unaryProcessBuilder;
    private Processor.BinaryProcess.Builder binaryProcessBuilder;
    private StorageBasic.StorageLocator.Builder storageLocatorBuilder;
    private Processor.TaskInfo.Builder taskInfoBuilder;
    private Processor.UnaryProcess defaultUnaryProcess;
    private Processor.BinaryProcess defaultBinaryProcess;
    private String name = "api_create_name";
    private String namespace = "api_create_namespace";

    public TestRollProcessServiceClient() {
        unaryProcessBuilder = Processor.UnaryProcess.newBuilder();
        binaryProcessBuilder = Processor.BinaryProcess.newBuilder();
        storageLocatorBuilder = StorageBasic.StorageLocator.newBuilder();
        taskInfoBuilder = Processor.TaskInfo.newBuilder();

        taskInfoBuilder.setFunctionId("funcid");
        storageLocatorBuilder.setType(StorageBasic.StorageType.LMDB)
                .setNamespace(namespace)
                .setName(name);
        StorageBasic.StorageLocator one = storageLocatorBuilder.build();
        StorageBasic.StorageLocator two = storageLocatorBuilder.setName(name + "_recv").build();

        defaultUnaryProcess = unaryProcessBuilder.setInfo(taskInfoBuilder).setOperand(one).build();
        defaultBinaryProcess = binaryProcessBuilder.setInfo(taskInfoBuilder).setLeft(one).setRight(two).build();
    }

    @Test
    public void testMap() throws Throwable {
        DelayedResult<StorageBasic.StorageLocator> delayedResult = rollProcessServiceClient.map(defaultUnaryProcess);

        processDelayedResult(delayedResult, "map");
    }

    @Test
    public void testMapValues() throws Throwable {
        DelayedResult<StorageBasic.StorageLocator> delayedResult = rollProcessServiceClient.mapValues(defaultUnaryProcess);

        processDelayedResult(delayedResult, "mapValues");
    }

    @Test
    public void testJoin() throws Throwable {
        DelayedResult<StorageBasic.StorageLocator> delayedResult = rollProcessServiceClient.join(defaultBinaryProcess);

        processDelayedResult(delayedResult, "join");
    }

    @Test
    public void testReduce() throws Throwable {
        OperandBroker operandBroker = rollProcessServiceClient.reduce(defaultUnaryProcess);

        Kv.Operand result = null;
        while (!operandBroker.isFinished()) {
            result = operandBroker.get();
            LOGGER.info("result: {}", toStringUtils.toOneLineString(result));
        }
    }

    @Test
    public void testMapPartitions() throws Throwable {
        DelayedResult<StorageBasic.StorageLocator> delayedResult = rollProcessServiceClient.mapPartitions(defaultUnaryProcess);

        processDelayedResult(delayedResult, "mapPartitions");
    }

    @Test
    public void testGlom() throws Throwable {
        DelayedResult<StorageBasic.StorageLocator> delayedResult = rollProcessServiceClient.glom(defaultUnaryProcess);

        processDelayedResult(delayedResult, "glom");
    }

    private void processDelayedResult(DelayedResult<? extends Message> delayedResult, String processName) throws Throwable {
        while (!delayedResult.getLatch().await(1, TimeUnit.SECONDS)) {
            ;
        }
        Message result = null;
        if (delayedResult.hasResult()) {
            result = delayedResult.getResultNow();
            LOGGER.info("result: {}", toStringUtils.toOneLineString(result));
        }

        if (delayedResult.hasError()) {
            Throwable t = delayedResult.getError();
            throw t;
        }

        if (processName == null) {
            processName = "unknown process";
        }
        LOGGER.info("{} finished", processName);
    }
}
