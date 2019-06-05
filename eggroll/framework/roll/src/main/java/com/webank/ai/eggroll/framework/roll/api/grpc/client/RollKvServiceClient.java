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
import com.webank.ai.eggroll.api.storage.KVServiceGrpc;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcAsyncClientContext;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcStreamingClientTemplate;
import com.webank.ai.eggroll.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.eggroll.core.constant.MetaConstants;
import com.webank.ai.eggroll.core.constant.ModelConstants;
import com.webank.ai.eggroll.core.constant.RuntimeConstants;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.model.DelayedResult;
import com.webank.ai.eggroll.core.model.impl.SingleDelayedResult;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.core.utils.TypeConversionUtils;
import com.webank.ai.eggroll.framework.roll.api.grpc.observer.kv.roll.RollKvCreateResponseObserver;
import com.webank.ai.eggroll.framework.roll.api.grpc.observer.kv.roll.RollKvIterateResponseStreamObserver;
import com.webank.ai.eggroll.framework.roll.api.grpc.observer.kv.roll.RollKvOperandToOperandObserver;
import com.webank.ai.eggroll.framework.roll.api.grpc.observer.kv.roll.RollKvPutResponseObserver;
import com.webank.ai.eggroll.framework.roll.api.grpc.observer.kv.storage.StorageKvCountResponseObserver;
import com.webank.ai.eggroll.framework.roll.api.grpc.observer.kv.storage.StorageKvDestroyResponseObserver;
import com.webank.ai.eggroll.framework.roll.api.grpc.observer.kv.storage.StorageKvPutAllClientResponseStreamObserver;
import com.webank.ai.eggroll.framework.roll.api.grpc.processor.caller.RollKvPutAllRequestStreamProcessor;
import com.webank.ai.eggroll.framework.roll.factory.RollKvCallModelFactory;
import com.webank.ai.eggroll.framework.roll.factory.RollModelFactory;
import com.webank.ai.eggroll.framework.roll.service.model.OperandBroker;
import com.webank.ai.eggroll.framework.roll.util.RollServerUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.TimeUnit;

@Component
@Scope("prototype")
public class RollKvServiceClient {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private RollKvCallModelFactory rollKvCallModelFactory;
    @Autowired
    private RollModelFactory rollModelFactory;
    @Autowired
    private ErrorUtils errorUtils;
    @Autowired
    private StorageMetaClient storageMetaClient;
    @Autowired
    private RollServerUtils rollServerUtils;
    @Autowired
    private TypeConversionUtils typeConversionUtils;
    private BasicMeta.Endpoint rollEndpoint = null;

    @PostConstruct
    // todo: move rollEndpoint info to global
    public void init() {
        rollEndpoint = rollServerUtils.getRollEndpoint();
    }

    public void create(Kv.CreateTableInfo request) {
        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.CreateTableInfo, Kv.CreateTableInfo> context
                = rollKvCallModelFactory.createCreateTableContext();

        context.setLatchInitCount(1)
                .setEndpoint(rollEndpoint)
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(KVServiceGrpc.KVServiceStub::createIfAbsent)
                .setCallerStreamObserverClassAndArguments(RollKvCreateResponseObserver.class);

        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.CreateTableInfo, Kv.CreateTableInfo> template
                = rollKvCallModelFactory.createCreateTableTemplate();
        template.setGrpcAsyncClientContext(context);

        template.calleeStreamingRpc(request);
    }

    public void put(Kv.Operand request, StoreInfo storeInfo) {
        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Empty> context
                = rollKvCallModelFactory.createOperandToEmptyContext();

        context.setLatchInitCount(1)
                .setEndpoint(rollEndpoint)
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(KVServiceGrpc.KVServiceStub::put)
                .setCallerStreamObserverClassAndArguments(RollKvPutResponseObserver.class)
                .setGrpcMetadata(MetaConstants.createMetadataFromStoreInfo(storeInfo));

        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Empty> template
                = rollKvCallModelFactory.createOperandToEmptyTemplate();
        template.setGrpcAsyncClientContext(context);

        template.calleeStreamingRpc(request);
    }

    public Kv.Operand putIfAbsent(Kv.Operand request, StoreInfo storeInfo) {
        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Operand> context
                = rollKvCallModelFactory.createOperandToOperandContext();

        DelayedResult<Kv.Operand> delayedResult = new SingleDelayedResult<>();

        context.setLatchInitCount(1)
                .setEndpoint(rollEndpoint)
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(KVServiceGrpc.KVServiceStub::putIfAbsent)
                .setCallerStreamObserverClassAndArguments(RollKvOperandToOperandObserver.class, delayedResult)
                .setGrpcMetadata(MetaConstants.createMetadataFromStoreInfo(storeInfo));

        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Operand> template
                = rollKvCallModelFactory.createOperandToOperandTemplate();
        template.setGrpcAsyncClientContext(context);

        Kv.Operand result = null;
        try {
            result = template.calleeStreamingRpcWithImmediateDelayedResult(request, delayedResult);
        } catch (InvocationTargetException e) {
            throw new RuntimeException(e);
        }

        return result;
    }

    public void putAll(OperandBroker operandBroker, StoreInfo storeInfo) {
        boolean needReset = true;
        boolean hasError = false;
        int resetInterval = 2;
        int remaining = resetInterval;
        int resetCount = 0;

        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Empty> context = null;
        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Empty> template = null;
        try {
            while (!operandBroker.isClosable()) {
                // possible init
                if (needReset) {
                    LOGGER.info("[ROLL][PUTALL][MAINTASK] resetting in putAll main. resetCount: {}", ++resetCount);
                    context = rollKvCallModelFactory.createOperandToEmptyContext();

                    context.setLatchInitCount(1)
                            .setEndpoint(rollEndpoint)
                            .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                            .setCallerStreamingMethodInvoker(KVServiceGrpc.KVServiceStub::putAll)
                            .setCallerStreamObserverClassAndArguments(StorageKvPutAllClientResponseStreamObserver.class)
                            .setGrpcMetadata(MetaConstants.createMetadataFromStoreInfo(storeInfo))
                            .setRequestStreamProcessorClassAndArguments(RollKvPutAllRequestStreamProcessor.class, operandBroker);

                    template = rollKvCallModelFactory.createOperandToEmptyTemplate();
                    template.setGrpcAsyncClientContext(context);

                    template.initCallerStreamingRpc();

                    remaining = resetInterval;
                    needReset = false;
                }

                // wait for data and send
                operandBroker.awaitLatch(500, TimeUnit.MILLISECONDS);
                template.processCallerStreamingRpc();
                --remaining;

                // possible cleanup
                if (remaining <= 0) {
                    template.completeStreamingRpc();
                    needReset = true;
                }
            }
        } catch (Throwable e) {
            LOGGER.error("[ROLL][PUTALL][MAINTASK] error in putAll main task: {}", errorUtils.getStackTrace(e));
            template.errorCallerStreamingRpc(e);
            hasError = true;
        } finally {
            if (!needReset && !hasError) {
                template.completeStreamingRpc();
            }
            // todo: possible duplicate with DtableRecvConsumeAction
            operandBroker.setFinished();
        }
    }

    public Kv.Operand delete(Kv.Operand request, StoreInfo storeInfo) {
        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Operand> context
                = rollKvCallModelFactory.createOperandToOperandContext();

        DelayedResult<Kv.Operand> delayedResult = new SingleDelayedResult<>();

        context.setLatchInitCount(1)
                .setEndpoint(rollEndpoint)
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(KVServiceGrpc.KVServiceStub::delOne)
                .setCallerStreamObserverClassAndArguments(RollKvOperandToOperandObserver.class, delayedResult)
                .setGrpcMetadata(MetaConstants.createMetadataFromStoreInfo(storeInfo));

        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Operand> template
                = rollKvCallModelFactory.createOperandToOperandTemplate();
        template.setGrpcAsyncClientContext(context);

        Kv.Operand result = null;
        try {
            result = template.calleeStreamingRpcWithImmediateDelayedResult(request, delayedResult);
        } catch (InvocationTargetException e) {
            throw new RuntimeException(e);
        }

        return result;
    }

    public Kv.Operand get(Kv.Operand request, StoreInfo storeInfo) {
        DelayedResult<Kv.Operand> delayedResult = new SingleDelayedResult<>();

        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Operand> context
                = rollKvCallModelFactory.createOperandToOperandContext();
        context.setLatchInitCount(1)
                .setEndpoint(rollEndpoint)
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(KVServiceGrpc.KVServiceStub::get)
                .setCallerStreamObserverClassAndArguments(RollKvOperandToOperandObserver.class, delayedResult)
                .setGrpcMetadata(MetaConstants.createMetadataFromStoreInfo(storeInfo));

        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Operand> template
                = rollKvCallModelFactory.createOperandToOperandTemplate();
        template.setGrpcAsyncClientContext(context);

        Kv.Operand result = null;
        try {
            result = template.calleeStreamingRpcWithImmediateDelayedResult(request, delayedResult);
        } catch (InvocationTargetException e) {
            throw new RuntimeException(e);
        }

        return result;
    }

    public OperandBroker iterate(Kv.Range request, StoreInfo storeInfo) {
        OperandBroker result = rollModelFactory.createOperandBroker();

        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Range, Kv.Operand> context
                = rollKvCallModelFactory.createRangeToOperandContext();
        context.setLatchInitCount(1)
                .setEndpoint(rollEndpoint)
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(KVServiceGrpc.KVServiceStub::iterate)
                .setCallerStreamObserverClassAndArguments(RollKvIterateResponseStreamObserver.class, result)
                .setGrpcMetadata(MetaConstants.createMetadataFromStoreInfo(storeInfo));

        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Range, Kv.Operand> template
                = rollKvCallModelFactory.createRangeToOperandTemplate();
        template.setGrpcAsyncClientContext(context);

        template.calleeStreamingRpc(request);

        return result;
    }

    public void destroy(StoreInfo storeInfo) {
        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Empty> context
                = rollKvCallModelFactory.createEmptyToEmptyContext();

        context.setLatchInitCount(1)
                .setEndpoint(rollEndpoint)
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(KVServiceGrpc.KVServiceStub::destroy)
                .setCallerStreamObserverClassAndArguments(StorageKvDestroyResponseObserver.class)
                .setGrpcMetadata(MetaConstants.createMetadataFromStoreInfo(storeInfo));

        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Empty> template
                = rollKvCallModelFactory.createEmptyToEmptyTemplate();
        template.setGrpcAsyncClientContext(context);

        template.calleeStreamingRpc(ModelConstants.EMPTY);
    }

    public void destroyAll(StoreInfo storeInfo) {
        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Empty> context
                = rollKvCallModelFactory.createEmptyToEmptyContext();

        context.setLatchInitCount(1)
                .setEndpoint(rollEndpoint)
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(KVServiceGrpc.KVServiceStub::destroyAll)
                .setCallerStreamObserverClassAndArguments(StorageKvDestroyResponseObserver.class)
                .setGrpcMetadata(MetaConstants.createMetadataFromStoreInfo(storeInfo));

        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Empty> template
                = rollKvCallModelFactory.createEmptyToEmptyTemplate();
        template.setGrpcAsyncClientContext(context);

        template.calleeStreamingRpc(ModelConstants.EMPTY);
    }

    public Kv.Count count(StoreInfo storeInfo) {
        DelayedResult<Kv.Count> delayedResult = new SingleDelayedResult<>();
        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Count> context
                = rollKvCallModelFactory.createEmptyToCountContext();

        context.setLatchInitCount(1)
                .setEndpoint(rollEndpoint)
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(KVServiceGrpc.KVServiceStub::count)
                .setCallerStreamObserverClassAndArguments(StorageKvCountResponseObserver.class, delayedResult)
                .setGrpcMetadata(MetaConstants.createMetadataFromStoreInfo(storeInfo));

        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Count> template
                = rollKvCallModelFactory.createEmptyToCountTemplate();
        template.setGrpcAsyncClientContext(context);

        Kv.Count result = null;

        try {
            result = template.calleeStreamingRpcWithImmediateDelayedResult(ModelConstants.EMPTY, delayedResult);
        } catch (InvocationTargetException e) {
            LOGGER.error(errorUtils.getStackTrace(e));

            throw new RuntimeException(e);
        }

        return result;
    }

    public BasicMeta.Endpoint getRollEndpoint() {
        return rollEndpoint;
    }

    public RollKvServiceClient setRollEndpoint(BasicMeta.Endpoint rollEndpoint) {
        this.rollEndpoint = rollEndpoint;
        return this;
    }
}
