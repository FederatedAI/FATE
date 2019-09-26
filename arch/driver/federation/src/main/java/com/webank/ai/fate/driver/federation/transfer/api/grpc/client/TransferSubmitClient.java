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

import com.google.common.base.Preconditions;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.fate.api.driver.federation.TransferSubmitServiceGrpc;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcAsyncClientContext;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcCalleeStreamingStubMethodInvoker;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcStreamingClientTemplate;
import com.webank.ai.eggroll.core.model.DelayedResult;
import com.webank.ai.eggroll.core.model.impl.SingleDelayedResult;
import com.webank.ai.fate.driver.federation.factory.TransferServiceFactory;
import com.webank.ai.fate.driver.federation.transfer.api.grpc.observer.SendClientResponseObserver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.TimeUnit;

@Component
@Scope("prototype")
public class TransferSubmitClient {
    @Autowired
    private TransferServiceFactory transferServiceFactory;

    public Federation.TransferMeta send(Federation.TransferMeta request, BasicMeta.Endpoint endpoint) {
        Preconditions.checkState(request.getType() == Federation.TransferType.SEND);
        return doTransferSubmitInternal(request, endpoint, TransferSubmitServiceGrpc.TransferSubmitServiceStub::send);
    }

    public Federation.TransferMeta recv(Federation.TransferMeta request, BasicMeta.Endpoint endpoint) {
        Preconditions.checkState(request.getType() == Federation.TransferType.RECV);
        return doTransferSubmitInternal(request, endpoint, TransferSubmitServiceGrpc.TransferSubmitServiceStub::recv);
    }

    public Federation.TransferMeta checkStatusNow(Federation.TransferMeta request, BasicMeta.Endpoint endpoint) {
        return doTransferSubmitInternal(request, endpoint, TransferSubmitServiceGrpc.TransferSubmitServiceStub::checkStatusNow);
    }

    public Federation.TransferMeta checkStatus(Federation.TransferMeta request, BasicMeta.Endpoint endpoint) {
        return doTransferSubmitInternal(request, endpoint, TransferSubmitServiceGrpc.TransferSubmitServiceStub::checkStatus);
    }

    private Federation.TransferMeta
    doTransferSubmitInternal(Federation.TransferMeta request,
                             BasicMeta.Endpoint endpoint,
                             GrpcCalleeStreamingStubMethodInvoker<
                                     TransferSubmitServiceGrpc.TransferSubmitServiceStub,
                                     Federation.TransferMeta,
                                     Federation.TransferMeta> calleeStreamingStubMethodInvoker) {
        DelayedResult<Federation.TransferMeta> delayedResult = new SingleDelayedResult<>();

        GrpcAsyncClientContext<TransferSubmitServiceGrpc.TransferSubmitServiceStub, Federation.TransferMeta, Federation.TransferMeta> asyncClientContext
                = transferServiceFactory.createSubmitClientGrpcAsyncClientContext();

        asyncClientContext.setLatchInitCount(1)
                .setEndpoint(endpoint)
                .setFinishTimeout(10, TimeUnit.MINUTES)
                .setCalleeStreamingMethodInvoker(calleeStreamingStubMethodInvoker)
                .setCallerStreamObserverClassAndArguments(SendClientResponseObserver.class, delayedResult);

        GrpcStreamingClientTemplate<TransferSubmitServiceGrpc.TransferSubmitServiceStub, Federation.TransferMeta, Federation.TransferMeta> template
                = transferServiceFactory.createSubmitClientTemplate();
        template.setGrpcAsyncClientContext(asyncClientContext);

        Federation.TransferMeta result = null;
        try {
            result = template.calleeStreamingRpcWithImmediateDelayedResult(request, delayedResult);
        } catch (InvocationTargetException e) {
            throw new RuntimeException(e);
        }

        return result;
    }


}
