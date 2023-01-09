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
package com.osx.broker.grpc;

import com.osx.broker.service.PushService;
import com.osx.broker.service.UnaryCallService;
import com.osx.broker.util.ContextUtil;
import com.osx.core.context.Context;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.OutboundPackage;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class ProxyGrpcService extends DataTransferServiceGrpc.DataTransferServiceImplBase {
    Logger logger = LoggerFactory.getLogger(ProxyGrpcService.class);
    UnaryCallService unaryCallService;
    PushService pushService;
    public ProxyGrpcService(PushService pushService,
                            UnaryCallService unaryCallService
    ) {
        this.pushService = pushService;
        this.unaryCallService = unaryCallService;
    }

    public io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> push(
            io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata> responseObserver) {
        try {
            Context context = ContextUtil.buildContext();
            InboundPackage<PushRequestDataWrap> data = new InboundPackage<>();
            PushRequestDataWrap pushRequestDataWrap = new PushRequestDataWrap();
            pushRequestDataWrap.setStreamObserver(responseObserver);
            data.setBody(pushRequestDataWrap);
            OutboundPackage<StreamObserver> outboundPackage = pushService.service(context, data);
            return outboundPackage.getData();
        } catch (Exception e) {
            logger.error("push error",e);
        }
        return null;
    }


    public void unaryCall(com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet request,
                          io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> responseObserver) {
        Context context = ContextUtil.buildContext();
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
