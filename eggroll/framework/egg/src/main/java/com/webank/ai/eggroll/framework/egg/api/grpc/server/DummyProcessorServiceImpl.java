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

package com.webank.ai.eggroll.framework.egg.api.grpc.server;

import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.computing.processor.ProcessServiceGrpc;
import com.webank.ai.eggroll.api.computing.processor.Processor;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.api.storage.StorageBasic;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicInteger;

@Component
// @Scope("prototype")
public class DummyProcessorServiceImpl extends ProcessServiceGrpc.ProcessServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();

    private AtomicInteger mapCount = new AtomicInteger(0);
    private AtomicInteger reduceCount = new AtomicInteger(0);

    @Override
    public void map(Processor.UnaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("egg map request received, cur count: {}", mapCount.incrementAndGet());

        StorageBasic.StorageLocator storageLocator = request.getOperand();
        responseObserver.onNext(storageLocator);
        responseObserver.onCompleted();
    }

    @Override
    public void mapValues(Processor.UnaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("egg mapValues request received");

        StorageBasic.StorageLocator storageLocator = request.getOperand();
        responseObserver.onNext(storageLocator);
        responseObserver.onCompleted();
    }

    @Override
    public void join(Processor.BinaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("egg join request received");

        StorageBasic.StorageLocator storageLocator = request.getRight();
        responseObserver.onNext(storageLocator);
        responseObserver.onCompleted();
    }

    @Override
    public void reduce(Processor.UnaryProcess request, StreamObserver<Kv.Operand> responseObserver) {
        LOGGER.info("egg reduce request received");

        long now = System.currentTimeMillis();

        Kv.Operand.Builder operandBuilder = Kv.Operand.newBuilder();
        Kv.Operand operand1 = operandBuilder.setKey(ByteString.copyFromUtf8("k1_" + reduceCount.getAndIncrement())).setValue(ByteString.copyFromUtf8("v1")).build();
        Kv.Operand operand2 = operandBuilder.setKey(ByteString.copyFromUtf8("k2_" + reduceCount.getAndIncrement())).setValue(ByteString.copyFromUtf8("v2")).build();

        responseObserver.onNext(operand1);
        responseObserver.onNext(operand2);
        responseObserver.onCompleted();
    }

    @Override
    public void mapPartitions(Processor.UnaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("egg mapPartitions request received");

        StorageBasic.StorageLocator storageLocator = request.getOperand();
        responseObserver.onNext(storageLocator);
        responseObserver.onCompleted();
    }

    @Override
    public void glom(Processor.UnaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("egg glom request received");

        StorageBasic.StorageLocator storageLocator = request.getOperand();
        responseObserver.onNext(storageLocator);
        responseObserver.onCompleted();
    }
}
