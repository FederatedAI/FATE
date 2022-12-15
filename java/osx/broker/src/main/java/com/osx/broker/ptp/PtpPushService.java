package com.osx.broker.ptp;

import com.osx.core.context.Context;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.service.AbstractServiceAdaptor;
import com.osx.core.service.InboundPackage;

public class PtpPushService extends AbstractServiceAdaptor {
    @Override
    protected Object doService(Context context, InboundPackage data) {


        //QueuePushReqStreamObserver

        return null;
    }

    @Override
    protected Object transformExceptionInfo(Context context, ExceptionInfo exceptionInfo) {
        return null;
    }
}
