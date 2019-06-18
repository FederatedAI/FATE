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

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.computing.processor.ProcessServiceGrpc;
import com.webank.ai.eggroll.api.computing.processor.Processor;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcAsyncClientContext;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcCalleeStreamingStubMethodInvoker;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcStreamingClientTemplate;
import com.webank.ai.eggroll.core.constant.RuntimeConstants;
import com.webank.ai.eggroll.core.model.DelayedResult;
import com.webank.ai.eggroll.core.model.impl.SingleDelayedResult;
import com.webank.ai.eggroll.core.utils.TypeConversionUtils;
import com.webank.ai.eggroll.framework.roll.api.grpc.observer.processor.egg.EggProcessorReduceResponseStreamObserver;
import com.webank.ai.eggroll.framework.roll.api.grpc.observer.processor.roll.RollProcessorUnaryProcessToStorageLocatorResponseObserver;
import com.webank.ai.eggroll.framework.roll.factory.RollModelFactory;
import com.webank.ai.eggroll.framework.roll.factory.RollProcessorServiceCallModelFactory;
import com.webank.ai.eggroll.framework.roll.service.model.OperandBroker;
import com.webank.ai.eggroll.framework.roll.util.RollServerUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

@Component
@Scope("prototype")
public class RollProcessServiceClient {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private RollProcessorServiceCallModelFactory rollProcessorServiceCallModelFactory;
    @Autowired
    private TypeConversionUtils typeConversionUtils;
    @Autowired
    private RollModelFactory rollModelFactory;
    @Autowired
    private RollServerUtils rollServerUtils;
    private BasicMeta.Endpoint rollEndpoint = RuntimeConstants.getLocalEndpoint(8011);

    public DelayedResult<StorageBasic.StorageLocator> map(Processor.UnaryProcess request) {
        LOGGER.info("roll map request received");

        return unaryProcessToStorageLocatorUnaryCall(request, ProcessServiceGrpc.ProcessServiceStub::map);
    }

    public DelayedResult<StorageBasic.StorageLocator> mapValues(Processor.UnaryProcess request) {
        LOGGER.info("roll mapValues request received");

        return unaryProcessToStorageLocatorUnaryCall(request, ProcessServiceGrpc.ProcessServiceStub::mapValues);
    }

    public DelayedResult<StorageBasic.StorageLocator> join(Processor.BinaryProcess request) {
        LOGGER.info("roll join request received");

        return binaryProcessToStorageLocatorUnaryCall(request, ProcessServiceGrpc.ProcessServiceStub::join);
    }

    public OperandBroker reduce(Processor.UnaryProcess request) {
        OperandBroker result = rollModelFactory.createOperandBroker();
        GrpcAsyncClientContext<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, Kv.Operand> context
                = rollProcessorServiceCallModelFactory.createUnaryProcessToOperandContext();

        context.setLatchInitCount(1)
                .setEndpoint(rollEndpoint)
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(ProcessServiceGrpc.ProcessServiceStub::reduce)
                .setCallerStreamObserverClassAndArguments(EggProcessorReduceResponseStreamObserver.class, result);

        GrpcStreamingClientTemplate<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, Kv.Operand> template
                = rollProcessorServiceCallModelFactory.createUnaryProcessToOperandTemplate();
        template.setGrpcAsyncClientContext(context);

        template.calleeStreamingRpc(request);

        return result;
    }

    public DelayedResult<StorageBasic.StorageLocator> mapPartitions(Processor.UnaryProcess request) {
        LOGGER.info("roll mapPartitions request received");

        return unaryProcessToStorageLocatorUnaryCall(request, ProcessServiceGrpc.ProcessServiceStub::mapPartitions);
    }

    public DelayedResult<StorageBasic.StorageLocator> glom(Processor.UnaryProcess request) {
        LOGGER.info("roll glom request received");

        return unaryProcessToStorageLocatorUnaryCall(request, ProcessServiceGrpc.ProcessServiceStub::glom);
    }

    public DelayedResult<StorageBasic.StorageLocator> subtractByKey(Processor.BinaryProcess request) {
        LOGGER.info("roll subtractByKey request received");

        return binaryProcessToStorageLocatorUnaryCall(request, ProcessServiceGrpc.ProcessServiceStub::subtractByKey);
    }

    public DelayedResult<StorageBasic.StorageLocator> filter(Processor.UnaryProcess request) {
        LOGGER.info("roll filter request received");

        return unaryProcessToStorageLocatorUnaryCall(request, ProcessServiceGrpc.ProcessServiceStub::filter);
    }

    public DelayedResult<StorageBasic.StorageLocator> union(Processor.BinaryProcess request) {
        LOGGER.info("roll union request received");

        return binaryProcessToStorageLocatorUnaryCall(request, ProcessServiceGrpc.ProcessServiceStub::union);
    }

    public DelayedResult<StorageBasic.StorageLocator> flatMap(Processor.UnaryProcess request) {
        LOGGER.info("roll flatMap request received");

        return unaryProcessToStorageLocatorUnaryCall(request, ProcessServiceGrpc.ProcessServiceStub::flatMap);
    }

    private DelayedResult<StorageBasic.StorageLocator>
    unaryProcessToStorageLocatorUnaryCall(Processor.UnaryProcess request,
                                          GrpcCalleeStreamingStubMethodInvoker<
                                                  ProcessServiceGrpc.ProcessServiceStub,
                                                  Processor.UnaryProcess,
                                                  StorageBasic.StorageLocator> calleeStreamingStubMethodInvoker) {
        GrpcAsyncClientContext<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, StorageBasic.StorageLocator> context
                = rollProcessorServiceCallModelFactory.createUnaryProcessToStorageLocatorContext();

        DelayedResult<StorageBasic.StorageLocator> delayedResult = new SingleDelayedResult<>();

        context.setLatchInitCount(1)
                .setEndpoint(rollEndpoint)
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(calleeStreamingStubMethodInvoker)
                .setCallerStreamObserverClassAndArguments(RollProcessorUnaryProcessToStorageLocatorResponseObserver.class, delayedResult);

        GrpcStreamingClientTemplate<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, StorageBasic.StorageLocator> template
                = rollProcessorServiceCallModelFactory.createUnaryProcessToStorageLocatorTemplate();
        template.setGrpcAsyncClientContext(context);

        template.calleeStreamingRpc(request);

        return delayedResult;
    }

    private DelayedResult<StorageBasic.StorageLocator>
    binaryProcessToStorageLocatorUnaryCall(Processor.BinaryProcess request,
                                          GrpcCalleeStreamingStubMethodInvoker<
                                                  ProcessServiceGrpc.ProcessServiceStub,
                                                  Processor.BinaryProcess,
                                                  StorageBasic.StorageLocator> calleeStreamingStubMethodInvoker) {
        GrpcAsyncClientContext<ProcessServiceGrpc.ProcessServiceStub, Processor.BinaryProcess, StorageBasic.StorageLocator> context
                = rollProcessorServiceCallModelFactory.createBinaryProcessToStorageLocatorContext();

        DelayedResult<StorageBasic.StorageLocator> delayedResult = new SingleDelayedResult<>();

        context.setLatchInitCount(1)
                .setEndpoint(rollEndpoint)
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(calleeStreamingStubMethodInvoker)
                .setCallerStreamObserverClassAndArguments(RollProcessorUnaryProcessToStorageLocatorResponseObserver.class, delayedResult);

        GrpcStreamingClientTemplate<ProcessServiceGrpc.ProcessServiceStub, Processor.BinaryProcess, StorageBasic.StorageLocator> template
                = rollProcessorServiceCallModelFactory.createBinaryProcessToStorageLocatorTemplate();
        template.setGrpcAsyncClientContext(context);

        template.calleeStreamingRpc(request);

        return delayedResult;
    }
}
