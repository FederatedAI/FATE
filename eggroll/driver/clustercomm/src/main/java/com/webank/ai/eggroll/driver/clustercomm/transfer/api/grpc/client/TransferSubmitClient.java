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

package com.webank.ai.eggroll.driver.clustercomm.transfer.api.grpc.client;

import com.google.common.base.Preconditions;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm;
import com.webank.ai.eggroll.api.driver.clustercomm.TransferSubmitServiceGrpc;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcAsyncClientContext;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcCalleeStreamingStubMethodInvoker;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcStreamingClientTemplate;
import com.webank.ai.eggroll.core.model.DelayedResult;
import com.webank.ai.eggroll.core.model.impl.SingleDelayedResult;
import com.webank.ai.eggroll.driver.clustercomm.factory.TransferServiceFactory;
import com.webank.ai.eggroll.driver.clustercomm.transfer.api.grpc.observer.SendClientResponseObserver;
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

    public ClusterComm.TransferMeta send(ClusterComm.TransferMeta request, BasicMeta.Endpoint endpoint) {
        Preconditions.checkState(request.getType() == ClusterComm.TransferType.SEND);
        return doTransferSubmitInternal(request, endpoint, TransferSubmitServiceGrpc.TransferSubmitServiceStub::send);
    }

    public ClusterComm.TransferMeta recv(ClusterComm.TransferMeta request, BasicMeta.Endpoint endpoint) {
        Preconditions.checkState(request.getType() == ClusterComm.TransferType.RECV);
        return doTransferSubmitInternal(request, endpoint, TransferSubmitServiceGrpc.TransferSubmitServiceStub::recv);
    }

    public ClusterComm.TransferMeta checkStatusNow(ClusterComm.TransferMeta request, BasicMeta.Endpoint endpoint) {
        return doTransferSubmitInternal(request, endpoint, TransferSubmitServiceGrpc.TransferSubmitServiceStub::checkStatusNow);
    }

    public ClusterComm.TransferMeta checkStatus(ClusterComm.TransferMeta request, BasicMeta.Endpoint endpoint) {
        return doTransferSubmitInternal(request, endpoint, TransferSubmitServiceGrpc.TransferSubmitServiceStub::checkStatus);
    }

    private ClusterComm.TransferMeta
    doTransferSubmitInternal(ClusterComm.TransferMeta request,
                             BasicMeta.Endpoint endpoint,
                             GrpcCalleeStreamingStubMethodInvoker<
                                     TransferSubmitServiceGrpc.TransferSubmitServiceStub,
                                     ClusterComm.TransferMeta,
                                     ClusterComm.TransferMeta> calleeStreamingStubMethodInvoker) {
        DelayedResult<ClusterComm.TransferMeta> delayedResult = new SingleDelayedResult<>();

        GrpcAsyncClientContext<TransferSubmitServiceGrpc.TransferSubmitServiceStub, ClusterComm.TransferMeta, ClusterComm.TransferMeta> asyncClientContext
                = transferServiceFactory.createSubmitClientGrpcAsyncClientContext();

        asyncClientContext.setLatchInitCount(1)
                .setEndpoint(endpoint)
                .setFinishTimeout(10, TimeUnit.MINUTES)
                .setCalleeStreamingMethodInvoker(calleeStreamingStubMethodInvoker)
                .setCallerStreamObserverClassAndArguments(SendClientResponseObserver.class, delayedResult);

        GrpcStreamingClientTemplate<TransferSubmitServiceGrpc.TransferSubmitServiceStub, ClusterComm.TransferMeta, ClusterComm.TransferMeta> template
                = transferServiceFactory.createSubmitClientTemplate();
        template.setGrpcAsyncClientContext(asyncClientContext);

        ClusterComm.TransferMeta result = null;
        try {
            result = template.calleeStreamingRpcWithImmediateDelayedResult(request, delayedResult);
        } catch (InvocationTargetException e) {
            throw new RuntimeException(e);
        }

        return result;
    }


}
