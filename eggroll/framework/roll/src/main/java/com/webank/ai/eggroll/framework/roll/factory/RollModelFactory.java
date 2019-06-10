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

package com.webank.ai.eggroll.framework.roll.factory;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import com.webank.ai.eggroll.framework.roll.api.grpc.client.EggProcessServiceClient;
import com.webank.ai.eggroll.framework.roll.service.async.callback.CountProcessListenableFutureCallback;
import com.webank.ai.eggroll.framework.roll.service.async.callback.DefaultRollProcessListenableFutureCallback;
import com.webank.ai.eggroll.framework.roll.service.async.callback.PutAllProcessorListenableFutureCallback;
import com.webank.ai.eggroll.framework.roll.service.async.processor.BaseProcessServiceProcessor;
import com.webank.ai.eggroll.framework.roll.service.async.storage.CountProcessor;
import com.webank.ai.eggroll.framework.roll.service.async.storage.IterateProcessor;
import com.webank.ai.eggroll.framework.roll.service.async.storage.PutAllProcessor;
import com.webank.ai.eggroll.framework.roll.service.handler.impl.ProcessServiceStorageLocatorResultHandler;
import com.webank.ai.eggroll.framework.roll.service.handler.impl.ReduceProcessServiceResultHandler;
import com.webank.ai.eggroll.framework.roll.service.model.OperandBroker;
import com.webank.ai.eggroll.framework.roll.service.model.OperandBrokerSortedHub;
import com.webank.ai.eggroll.framework.roll.service.model.OperandBrokerUnSortedHub;
import com.webank.ai.eggroll.framework.roll.strategy.DispatchPolicy;
import com.webank.ai.eggroll.framework.roll.strategy.impl.DefaultModDispatchPolicy;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class RollModelFactory {
    @Autowired
    private ApplicationContext applicationContext;

    public OperandBroker createOperandBroker() {
        return applicationContext.getBean(OperandBroker.class);
    }

    public OperandBroker createOperandBroker(int capacity) {
        return applicationContext.getBean(OperandBroker.class, capacity);
    }

    public OperandBrokerSortedHub createOperandSortedBrokerHub() {
        return applicationContext.getBean(OperandBrokerSortedHub.class);
    }

    public OperandBrokerUnSortedHub createOperandUnsortedBrokerHub() {
        return applicationContext.getBean(OperandBrokerUnSortedHub.class);
    }

    public PutAllProcessor createPutAllProcessor(OperandBroker operandBroker, StoreInfo storeInfo, Node node) {
        return applicationContext.getBean(PutAllProcessor.class, operandBroker, storeInfo, node);
    }

    public IterateProcessor createIterateProcessor(Kv.Range range, StoreInfo storeInfo, final OperandBroker operandBroker) {
        return applicationContext.getBean(IterateProcessor.class, range, storeInfo, operandBroker);
    }

    public CountProcessor createCountProcessor(Kv.Empty request, StoreInfo storeInfo, Node node) {
        return applicationContext.getBean(CountProcessor.class, request, storeInfo, node);
    }

    public <R, E> BaseProcessServiceProcessor<R, E> createBaseProcessServiceProcessor(Class<? extends BaseProcessServiceProcessor<R, E>> concreteProcessServiceProcessorClass,
                                                                                      EggProcessServiceClient eggProcessServiceClient,
                                                                                      R requestInstance,
                                                                                      BasicMeta.Endpoint processorEndpoint) {
        return applicationContext.getBean(concreteProcessServiceProcessorClass, eggProcessServiceClient, requestInstance, processorEndpoint);
    }

    public ProcessServiceStorageLocatorResultHandler createProcessServiceStorageLocatorResultHandler() {
        return applicationContext.getBean(ProcessServiceStorageLocatorResultHandler.class);
    }

    public ReduceProcessServiceResultHandler createReduceProcessServiceResultHandler() {
        return applicationContext.getBean(ReduceProcessServiceResultHandler.class);
    }

    public <T> DefaultRollProcessListenableFutureCallback<T> createDefaultRollProcessListenableCallback(final List<T> resultContainer,
                                                                                                        final List<Throwable> errorContainer,
                                                                                                        final CountDownLatch finishLatch,
                                                                                                        final String ip,
                                                                                                        final int port) {
        return (DefaultRollProcessListenableFutureCallback<T>) applicationContext.getBean(
                DefaultRollProcessListenableFutureCallback.class,
                resultContainer,
                errorContainer,
                finishLatch,
                ip,
                port);
    }

    public CountProcessListenableFutureCallback createCountProcessListenableFutureCallback(final AtomicLong countResult,
                                                                                           final List<Throwable> errorContainer,
                                                                                           final CountDownLatch finishLatch,
                                                                                           final String ip,
                                                                                           final int port) {
        return applicationContext.getBean(
                CountProcessListenableFutureCallback.class,
                countResult,
                errorContainer,
                finishLatch,
                ip,
                port);
    }


    public PutAllProcessorListenableFutureCallback createPutAllProcessorListenableFutureCallback(List<BasicMeta.ReturnStatus> resultContainer,
                                                                                                 List<Throwable> errorContainer,
                                                                                                 CountDownLatch eggPutAllFinishLatch,
                                                                                                 String ip,
                                                                                                 Integer port,
                                                                                                 StoreInfo storeInfoWithFragment,
                                                                                                 Set<Integer> finishedFragmentSet) {
        return applicationContext.getBean(PutAllProcessorListenableFutureCallback.class,
                resultContainer,
                errorContainer,
                eggPutAllFinishLatch,
                ip,
                port,
                storeInfoWithFragment,
                finishedFragmentSet);
    }

    public DispatchPolicy createDefaultDispatchPolicy() {
        return applicationContext.getBean(DefaultModDispatchPolicy.class);
    }
}