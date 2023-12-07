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
package org.fedai.osx.broker.service;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import io.grpc.stub.StreamObserver;
import org.fedai.osx.broker.grpc.QueuePushReqStreamObserver;
import org.fedai.osx.broker.queue.TransferQueueManager;
import org.fedai.osx.broker.router.RouterServiceRegister;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.exceptions.SysException;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.fedai.osx.core.service.InboundPackage;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Singleton
public class PushService extends AbstractServiceAdaptorNew<InboundPackage<StreamObserver>, StreamObserver> {

    Logger logger = LoggerFactory.getLogger(PushService.class);
    @Inject
    RouterServiceRegister  routerServiceRegister;
    @Inject
    TransferQueueManager transferQueueManager;

    @Override
    protected StreamObserver doService(OsxContext context, InboundPackage<StreamObserver> data
    ) {

        StreamObserver backRespSO = data.getBody();
        // context.setNeedPrintFlowLog(false);
        QueuePushReqStreamObserver queuePushReqStreamObserver = new QueuePushReqStreamObserver(context,
                routerServiceRegister.select(MetaInfo.PROPERTY_FATE_TECH_PROVIDER), transferQueueManager,
                backRespSO);
        return queuePushReqStreamObserver;
    }

    @Override
    protected StreamObserver transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        logger.error("PushService error {}", exceptionInfo);
        throw new SysException(exceptionInfo.toString());
    }

    @Override
    public InboundPackage<StreamObserver> decode(Object object) {
        return null;
    }

    @Override
    public Osx.Outbound toOutbound(StreamObserver response) {
        return null;
    }
}
