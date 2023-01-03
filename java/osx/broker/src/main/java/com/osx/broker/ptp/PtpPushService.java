package com.osx.broker.ptp;

import com.osx.broker.grpc.PushRequestDataWrap;
import com.osx.broker.grpc.QueuePushReqStreamObserver;
import com.osx.broker.util.TransferUtil;
import com.osx.core.context.Context;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.ptp.TargetMethod;
import com.osx.core.service.AbstractServiceAdaptor;
import com.osx.core.service.InboundPackage;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PtpPushService extends AbstractServiceAdaptor<StreamObserver, StreamObserver> {
    Logger  logger = LoggerFactory.getLogger(PtpPushService.class);

    @Override
    protected StreamObserver doService(Context context, InboundPackage<StreamObserver> data) {

        logger.info("================  receive PtpPushService");
        StreamObserver responseStreamObserver = data.getBody();
        return  new StreamObserver<Osx.Inbound>() {
            Logger logger = LoggerFactory.getLogger(PtpPushService.class);
            QueuePushReqStreamObserver queuePushReqStreamObserver = new  QueuePushReqStreamObserver(context,responseStreamObserver,Osx.Outbound.class);
            @Override
            public void onNext(Osx.Inbound inbound) {
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
    protected StreamObserver transformExceptionInfo(Context context, ExceptionInfo exceptionInfo) {
        return null;
    }
}
