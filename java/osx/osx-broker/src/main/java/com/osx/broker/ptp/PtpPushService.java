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
package com.osx.broker.ptp;
import com.osx.api.context.Context;
import com.osx.broker.ServiceContainer;

import com.osx.broker.grpc.QueuePushReqStreamObserver;
import com.osx.broker.util.TransferUtil;
import com.osx.core.config.MetaInfo;
import com.osx.core.context.FateContext;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.ptp.TargetMethod;
import com.osx.core.service.AbstractServiceAdaptor;
import com.osx.core.service.InboundPackage;
import com.osx.core.token.TokenResult;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
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
            QueuePushReqStreamObserver queuePushReqStreamObserver = new  QueuePushReqStreamObserver(context,ServiceContainer.routerRegister.getRouterService(MetaInfo.PROPERTY_FATE_TECH_PROVIDER),
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
