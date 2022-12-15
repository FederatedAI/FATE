package com.osx.broker.ptp;

import com.osx.core.context.Context;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.service.AbstractServiceAdaptor;
import org.ppc.ptp.Pcp;

public abstract class AbstractPtpServiceAdaptor extends AbstractServiceAdaptor<Pcp.Inbound, Pcp.Outbound> {

    @Override
    protected Pcp.Outbound transformExceptionInfo(Context context, ExceptionInfo exceptionInfo) {
        Pcp.Outbound.Builder  builder =  Pcp.Outbound.newBuilder();
        builder.setCode(exceptionInfo.getCode());
        builder.setMessage(exceptionInfo.getMessage());
        return  builder.build();
    }

}
