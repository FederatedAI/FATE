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

package com.webank.ai.fate.driver.federation.factory;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.fate.api.driver.federation.TransferSubmitServiceGrpc;
import com.webank.ai.fate.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcAsyncClientContext;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcStreamingClientTemplate;
import com.webank.ai.fate.driver.federation.transfer.api.grpc.observer.PushServerRequestStreamObserver;
import com.webank.ai.fate.driver.federation.transfer.communication.action.DtableRecvConsumeAction;
import com.webank.ai.fate.driver.federation.transfer.communication.action.ObjectRecvConsumeLmdbAction;
import com.webank.ai.fate.driver.federation.transfer.communication.action.SendConsumeAction;
import com.webank.ai.fate.driver.federation.transfer.communication.action.TransferQueueConsumeAction;
import com.webank.ai.fate.driver.federation.transfer.communication.consumer.TransferBrokerConsumer;
import com.webank.ai.fate.driver.federation.transfer.communication.processor.RecvProcessor;
import com.webank.ai.fate.driver.federation.transfer.communication.processor.SendProcessor;
import com.webank.ai.fate.driver.federation.transfer.communication.producer.DtableFragmentSendProducer;
import com.webank.ai.fate.driver.federation.transfer.communication.producer.ObjectLmdbSendProducer;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.Fragment;
import io.grpc.stub.StreamObserver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicBoolean;

@Component
public class TransferServiceFactory {
    @Autowired
    private ApplicationContext applicationContext;

    @Autowired
    private GrpcStreamingClientTemplate<DataTransferServiceGrpc.DataTransferServiceStub, Proxy.Packet, Proxy.Metadata> nonSpringPushTemplate;
    @Autowired
    private GrpcAsyncClientContext<DataTransferServiceGrpc.DataTransferServiceStub, Proxy.Packet, Proxy.Metadata> nonSpringPushContext;
    @Autowired
    private GrpcStreamingClientTemplate<DataTransferServiceGrpc.DataTransferServiceStub, Proxy.Packet, Proxy.Packet> nonSpringUnaryCallTemplate;
    @Autowired
    private GrpcAsyncClientContext<DataTransferServiceGrpc.DataTransferServiceStub, Proxy.Packet, Proxy.Packet> nonSpringUnaryCallContext;
    @Autowired
    private GrpcStreamingClientTemplate<TransferSubmitServiceGrpc.TransferSubmitServiceStub, Federation.TransferMeta, Federation.TransferMeta> nonSpringSubmitTemplate;
    @Autowired
    private GrpcAsyncClientContext<TransferSubmitServiceGrpc.TransferSubmitServiceStub, Federation.TransferMeta, Federation.TransferMeta> nonSpringSubmitContext;

    public TransferServiceFactory() {
    }

    public SendProcessor createSendProcessor(Federation.TransferMeta transferMeta) {
        return applicationContext.getBean(SendProcessor.class, transferMeta);
    }

    public RecvProcessor createRecvProcessor(Federation.TransferMeta transferMeta) {
        return applicationContext.getBean(RecvProcessor.class, transferMeta);
    }

    public DtableFragmentSendProducer createDtableFragmentSendProducer(Fragment fragment, TransferBroker broker) {
        return applicationContext.getBean(DtableFragmentSendProducer.class, fragment, broker);
    }

    public TransferBrokerConsumer createTransferBrokerConsumer() {
        return applicationContext.getBean(TransferBrokerConsumer.class);
    }

    public TransferBroker createTransferBroker(Federation.TransferMeta transferMeta) {
        return applicationContext.getBean(TransferBroker.class, transferMeta);
    }

    public TransferBroker createTransferBroker(Federation.TransferMeta transferMeta, int capacity) {
        return applicationContext.getBean(TransferBroker.class, transferMeta, capacity);
    }

    public TransferBroker createTransferBroker(String transferMetaId) {
        return applicationContext.getBean(TransferBroker.class, transferMetaId);
    }

    public TransferBroker createTransferBroker(String transferMetaId, int capacity) {
        return applicationContext.getBean(TransferBroker.class, transferMetaId, capacity);
    }

    public TransferQueueConsumeAction createSendConsumeAction(TransferBroker transferBroker, BasicMeta.Endpoint target) {
        return applicationContext.getBean(SendConsumeAction.class, transferBroker, target);
    }

/*    public PushStreamProcessor createPushStreamProcessor(TransferBroker transferBroker) {
        return applicationContext.getBean(PushStreamProcessor.class, transferBroker);
    }*/

    public GrpcAsyncClientContext<DataTransferServiceGrpc.DataTransferServiceStub, Proxy.Packet, Proxy.Metadata>
    createPushClientGrpcAsyncClientContext() {
        return applicationContext.getBean(nonSpringPushContext.getClass())
                .setStubClass(DataTransferServiceGrpc.DataTransferServiceStub.class);
    }

    public GrpcStreamingClientTemplate<DataTransferServiceGrpc.DataTransferServiceStub, Proxy.Packet, Proxy.Packet>
    createUnaryCallClientTemplate() {
        return applicationContext.getBean(nonSpringUnaryCallTemplate.getClass());
    }

    public GrpcAsyncClientContext<DataTransferServiceGrpc.DataTransferServiceStub, Proxy.Packet, Proxy.Packet>
    createUnaryCallClientGrpcAsyncClientContext() {
        return applicationContext.getBean(nonSpringUnaryCallContext.getClass())
                .setStubClass(DataTransferServiceGrpc.DataTransferServiceStub.class);
    }

    public GrpcStreamingClientTemplate<DataTransferServiceGrpc.DataTransferServiceStub, Proxy.Packet, Proxy.Metadata>
    createPushClientTemplate() {
        return applicationContext.getBean(nonSpringPushTemplate.getClass());
    }

    public GrpcAsyncClientContext<TransferSubmitServiceGrpc.TransferSubmitServiceStub, Federation.TransferMeta, Federation.TransferMeta>
    createSubmitClientGrpcAsyncClientContext() {
        return applicationContext.getBean(nonSpringSubmitContext.getClass())
                .setStubClass(TransferSubmitServiceGrpc.TransferSubmitServiceStub.class);
    }

    public GrpcStreamingClientTemplate<TransferSubmitServiceGrpc.TransferSubmitServiceStub, Federation.TransferMeta, Federation.TransferMeta>
    createSubmitClientTemplate() {
        return applicationContext.getBean(nonSpringSubmitTemplate.getClass());
    }

    public PushServerRequestStreamObserver createPushServerRequestStreamObserver(final StreamObserver<Proxy.Metadata> streamObserver,
                                                                                 final AtomicBoolean wasReady) {
        return applicationContext.getBean(PushServerRequestStreamObserver.class, streamObserver, wasReady);
    }

    public TransferQueueConsumeAction createDtableRecvConsumeAction(TransferBroker transferBroker) {
        return applicationContext.getBean(DtableRecvConsumeAction.class, transferBroker);
    }

    public TransferQueueConsumeAction createObjectRecvConsumeLmdbAction(TransferBroker transferBroker) {
        return applicationContext.getBean(ObjectRecvConsumeLmdbAction.class, transferBroker);
    }

    public ObjectLmdbSendProducer createLmdbSendProducer(TransferBroker transferBroker) {
        return applicationContext.getBean(ObjectLmdbSendProducer.class, transferBroker);
    }
}
