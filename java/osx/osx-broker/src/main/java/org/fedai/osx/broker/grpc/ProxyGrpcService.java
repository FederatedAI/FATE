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

import com.google.inject.Inject;
import com.google.inject.Singleton;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import org.fedai.osx.broker.ptp.EggrollPushService;
import org.fedai.osx.broker.service.PushService;
import org.fedai.osx.broker.service.UnaryCallService;
import org.fedai.osx.broker.util.ContextUtil;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.context.Protocol;
import org.fedai.osx.core.service.InboundPackage;
import org.fedai.osx.core.utils.FlowLogUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Singleton
public class ProxyGrpcService extends DataTransferServiceGrpc.DataTransferServiceImplBase {
    Logger logger = LoggerFactory.getLogger(ProxyGrpcService.class);
    @Inject
    UnaryCallService unaryCallService;
    @Inject
    PushService pushService;
    @Inject
    EggrollPushService eggrollPushService;

    public io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> push(
            io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata> responseObserver) {
        try {
            OsxContext context = ContextUtil.buildFateContext(Protocol.grpc);
            context.setNeedPrintFlowLog(false);
            InboundPackage<StreamObserver> data = new InboundPackage<>();
            data.setBody(responseObserver);
            StreamObserver result = pushService.service(context, data);
            return result;
        } catch (Exception e) {
            logger.error("push error", e);
        }
        return null;
    }


    public void unaryCall(com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet request,
                          io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> responseObserver) {
        OsxContext context = ContextUtil.buildFateContext(Protocol.grpc);
        try {

            InboundPackage<Proxy.Packet> data = new InboundPackage<>();
            data.setBody(request);
            context.setDataSize(request.getSerializedSize());
            Proxy.Packet result = unaryCallService.service(context, request);
            responseObserver.onNext(result);
            responseObserver.onCompleted();
        } catch (Exception e) {
            responseObserver.onError(e);
        } finally {
            FlowLogUtil.printFlowLog(context);
        }
    }


    public io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame> polling(
            io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame> responseObserver) {
        return null;
    }


}
