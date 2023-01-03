package com.osx.broker.ptp;

import com.osx.core.context.Context;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.service.AbstractServiceAdaptor;
import org.ppc.ptp.Osx;

public abstract class AbstractPtpServiceAdaptor extends AbstractServiceAdaptor<Osx.Inbound, Osx.Outbound> {

    @Override
    protected Osx.Outbound transformExceptionInfo(Context context, ExceptionInfo exceptionInfo) {
        Osx.Outbound.Builder builder = Osx.Outbound.newBuilder();
        builder.setCode(exceptionInfo.getCode());
        builder.setMessage(exceptionInfo.getMessage());
        return builder.build();
    }

}
