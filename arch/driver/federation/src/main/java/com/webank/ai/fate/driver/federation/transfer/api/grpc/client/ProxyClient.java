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

package com.webank.ai.fate.driver.federation.transfer.api.grpc.client;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.fate.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcAsyncClientContext;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcStreamingClientTemplate;
import com.webank.ai.eggroll.core.constant.RuntimeConstants;
import com.webank.ai.eggroll.core.model.DelayedResult;
import com.webank.ai.eggroll.core.model.impl.SingleDelayedResult;
import com.webank.ai.eggroll.core.server.DefaultServerConf;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.fate.driver.federation.factory.TransferServiceFactory;
import com.webank.ai.fate.driver.federation.transfer.api.grpc.observer.PushClientResponseStreamObserver;
import com.webank.ai.fate.driver.federation.transfer.api.grpc.observer.UnaryCallServerRequestStreamObserver;
import com.webank.ai.fate.driver.federation.transfer.api.grpc.processor.PushStreamProcessor;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import com.webank.ai.fate.driver.federation.transfer.utils.TransferProtoMessageUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.atomic.AtomicBoolean;

@Component
@Scope("prototype")
public class ProxyClient {
    @Autowired
    private TransferServiceFactory transferServiceFactory;
    @Autowired
    private TransferProtoMessageUtils transferProtoMessageUtils;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private DefaultServerConf defaultServerConf;

    private AtomicBoolean inited = new AtomicBoolean(false);

    private GrpcStreamingClientTemplate<DataTransferServiceGrpc.DataTransferServiceStub, Proxy.Packet, Proxy.Metadata> pushTemplate;
    private static final Logger LOGGER = LogManager.getLogger();

    public synchronized void initPush(TransferBroker request, BasicMeta.Endpoint endpoint) {
        LOGGER.info("[DEBUG][FEDERATION] initPush. broker: {}, transferMetaId: {}", request, toStringUtils.toOneLineString(request.getTransferMeta()));

        GrpcAsyncClientContext<DataTransferServiceGrpc.DataTransferServiceStub, Proxy.Packet, Proxy.Metadata> asyncClientContext
                = transferServiceFactory.createPushClientGrpcAsyncClientContext();

        asyncClientContext.setLatchInitCount(1)
                .setEndpoint(endpoint)
                .setSecureRequest(defaultServerConf.isSecureClient())
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCallerStreamingMethodInvoker(DataTransferServiceGrpc.DataTransferServiceStub::push)
                .setCallerStreamObserverClassAndArguments(PushClientResponseStreamObserver.class, request)
                .setRequestStreamProcessorClassAndArguments(PushStreamProcessor.class, request);

        pushTemplate = transferServiceFactory.createPushClientTemplate();
        pushTemplate.setGrpcAsyncClientContext(asyncClientContext);

        pushTemplate.initCallerStreamingRpc();

        inited.compareAndSet(false, true);
    }

    public void doPush() {
        if (pushTemplate == null) {
            throw new IllegalStateException("pushTemplate has not been initialized yet");
        }

        while (!inited.get()) {
            LOGGER.info("[DEBUG][FEDERATION] proxyClient not inited yet");

            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                LOGGER.error("error in doPush: " + ExceptionUtils.getStackTrace(e));
            }
        }
        pushTemplate.processCallerStreamingRpc();
    }

    public synchronized void completePush() {
        // LOGGER.info("[PUSH][CLIENT] completing push");
        if (pushTemplate == null) {
            throw new IllegalStateException("pushTemplate has not been initialized yet");
        }
        pushTemplate.completeStreamingRpc();
    }

    public Proxy.Packet unaryCall(Proxy.Packet request, BasicMeta.Endpoint endpoint) {
        DelayedResult<Proxy.Packet> delayedResult = new SingleDelayedResult<>();
        GrpcAsyncClientContext<DataTransferServiceGrpc.DataTransferServiceStub, Proxy.Packet, Proxy.Packet> context
                = transferServiceFactory.createUnaryCallClientGrpcAsyncClientContext();

        context.setLatchInitCount(1)
                .setEndpoint(endpoint)
                .setSecureRequest(defaultServerConf.isSecureClient())
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(DataTransferServiceGrpc.DataTransferServiceStub::unaryCall)
                .setCallerStreamObserverClassAndArguments(UnaryCallServerRequestStreamObserver.class, delayedResult);

        GrpcStreamingClientTemplate<DataTransferServiceGrpc.DataTransferServiceStub, Proxy.Packet, Proxy.Packet> template
                = transferServiceFactory.createUnaryCallClientTemplate();
        template.setGrpcAsyncClientContext(context);

        Proxy.Packet result = null;
        try {
            result = template.calleeStreamingRpcWithImmediateDelayedResult(request, delayedResult);
        } catch (InvocationTargetException e) {
            throw new RuntimeException(e);
        }

        return result;
    }

    public Federation.TransferMeta requestSendStart(Federation.TransferMeta transferMeta, BasicMeta.Endpoint endpoint) {


        Proxy.Packet request = transferProtoMessageUtils.generateSendStartRequest(transferMeta);

        Proxy.Packet response = unaryCall(request, endpoint);

        Federation.TransferMeta result = transferProtoMessageUtils.extractTransferMetaFromPacket(response);

        return result;
    }

    public Federation.TransferMeta requestSendEnd(Federation.TransferMeta transferMeta, BasicMeta.Endpoint endpoint) {
        Proxy.Packet request = transferProtoMessageUtils.generateSendEndRequest(transferMeta);

        Proxy.Packet response = unaryCall(request, endpoint);

        Federation.TransferMeta result = transferProtoMessageUtils.extractTransferMetaFromPacket(response);

        return result;
    }
}
