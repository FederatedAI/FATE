package com.osx.broker.service;

import com.osx.broker.grpc.PushRequestDataWrap;
import com.osx.broker.grpc.QueuePushReqStreamObserver;
import com.osx.core.context.Context;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.service.AbstractServiceAdaptor;
import com.osx.core.service.InboundPackage;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PushService2 extends AbstractServiceAdaptor<PushRequestDataWrap, StreamObserver> {


    Logger logger = LoggerFactory.getLogger(PushService2.class);

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
        throw new RuntimeException("xxxxx");
    }
}
