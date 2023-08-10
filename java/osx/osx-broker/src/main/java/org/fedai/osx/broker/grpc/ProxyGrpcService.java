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
package org.fedai.osx.broker.grpc;

import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import org.fedai.osx.api.constants.Protocol;
import org.fedai.osx.broker.interceptor.RouterInterceptor;
import org.fedai.osx.broker.interceptor.UnaryCallHandleInterceptor;
import org.fedai.osx.broker.service.PushService;
import org.fedai.osx.broker.service.UnaryCallService;
import org.fedai.osx.broker.util.ContextUtil;
import org.fedai.osx.core.context.FateContext;
import org.fedai.osx.core.service.InboundPackage;
import org.fedai.osx.core.service.OutboundPackage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class ProxyGrpcService extends DataTransferServiceGrpc.DataTransferServiceImplBase {
    Logger logger = LoggerFactory.getLogger(ProxyGrpcService.class);
    UnaryCallService unaryCallService;
    PushService pushService;
    public ProxyGrpcService(
    ) {
        this.pushService = new PushService();
        this.unaryCallService  =new UnaryCallService();
        unaryCallService .addPreProcessor(new UnaryCallHandleInterceptor()).
                addPreProcessor(new RouterInterceptor());

    }

    public io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> push(
            io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata> responseObserver) {
        try {
            FateContext context = ContextUtil.buildFateContext(Protocol.grpc);
            context.setNeedPrintFlowLog(false);
            InboundPackage<StreamObserver> data = new InboundPackage<>();
            data.setBody(responseObserver);
            OutboundPackage<StreamObserver> outboundPackage = pushService.service(context, data);
            return outboundPackage.getData();
        } catch (Exception e) {
            logger.error("push error",e);
        }
        return null;
    }


    public void unaryCall(com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet request,
                          io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> responseObserver) {
        FateContext context = ContextUtil.buildFateContext(Protocol.grpc);
        InboundPackage<Proxy.Packet> data = new InboundPackage<>();
        data.setBody(request);
        context.setDataSize(request.getSerializedSize());
        OutboundPackage<Proxy.Packet> outboundPackage = unaryCallService.service(context, data);
        Proxy.Packet result = outboundPackage.getData();
        Throwable throwable = outboundPackage.getThrowable();
        if (throwable != null) {
            responseObserver.onError(throwable);
        } else {
            responseObserver.onNext(result);
            responseObserver.onCompleted();
        }
    }


    public io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame> polling(
            io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame> responseObserver) {
        return null;
    }


}
