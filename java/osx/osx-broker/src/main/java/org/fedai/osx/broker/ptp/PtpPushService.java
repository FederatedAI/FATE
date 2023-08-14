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
package org.fedai.osx.broker.ptp;

import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import org.fedai.osx.broker.ServiceContainer;
import org.fedai.osx.broker.grpc.QueuePushReqStreamObserver;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.context.FateContext;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.service.AbstractServiceAdaptor;
import org.fedai.osx.core.service.InboundPackage;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class PtpPushService extends AbstractServiceAdaptor<FateContext,StreamObserver, StreamObserver> {
    Logger  logger = LoggerFactory.getLogger(PtpPushService.class);



    @Override
    protected StreamObserver doService(FateContext context, InboundPackage<StreamObserver> data) {
        StreamObserver responseStreamObserver = data.getBody();
        context.setNeedPrintFlowLog(false);
        return  new StreamObserver<Osx.Inbound>() {
            Logger logger = LoggerFactory.getLogger(PtpPushService.class);
            QueuePushReqStreamObserver queuePushReqStreamObserver = new  QueuePushReqStreamObserver(context, ServiceContainer.routerRegister.getRouterService(MetaInfo.PROPERTY_FATE_TECH_PROVIDER),
                    responseStreamObserver,Osx.Outbound.class);
            @Override
            public void onNext(Osx.Inbound inbound) {
                int dataSize = inbound.getSerializedSize();
                ServiceContainer.tokenApplyService.applyToken(context, TransferUtil.buildResource(inbound), dataSize);
                Proxy.Packet  packet =   TransferUtil.parsePacketFromInbound(inbound);
                if(packet!=null) {
                    queuePushReqStreamObserver.onNext(packet);
                }else{
                    logger.error("parse inbound error");
                }
            }
            @Override
            public void onError(Throwable throwable) {
                queuePushReqStreamObserver.onError(throwable);
            }
            @Override
            public void onCompleted() {
                queuePushReqStreamObserver.onCompleted();
            }
        };
    }

    @Override
    protected StreamObserver transformExceptionInfo(FateContext context, ExceptionInfo exceptionInfo) {
        return null;
    }
}
