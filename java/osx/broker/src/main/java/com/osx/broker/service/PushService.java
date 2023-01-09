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
package com.osx.broker.service;

import com.osx.broker.grpc.PushRequestDataWrap;
import com.osx.broker.grpc.QueuePushReqStreamObserver;
import com.osx.core.context.Context;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.exceptions.SysException;
import com.osx.core.service.AbstractServiceAdaptor;
import com.osx.core.service.InboundPackage;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PushService extends AbstractServiceAdaptor<PushRequestDataWrap, StreamObserver> {

    Logger logger = LoggerFactory.getLogger(PushService.class);
    
    @Override
    protected StreamObserver doService(Context context, InboundPackage<PushRequestDataWrap> data
    ) {

        PushRequestDataWrap pushRequestDataWrap = data.getBody();
        StreamObserver backRespSO = pushRequestDataWrap.getStreamObserver();
        context.setNeedPrintFlowLog(false);
        QueuePushReqStreamObserver queuePushReqStreamObserver = new QueuePushReqStreamObserver(context,
                backRespSO, Proxy.Metadata.class);
        return queuePushReqStreamObserver;
    }

    @Override
    protected StreamObserver transformExceptionInfo(Context context, ExceptionInfo exceptionInfo) {
        logger.error("PushService error {}", exceptionInfo);
        throw new SysException(exceptionInfo.toString());
    }
}
