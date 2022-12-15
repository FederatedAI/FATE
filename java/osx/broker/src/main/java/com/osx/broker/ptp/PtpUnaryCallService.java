package com.osx.broker.ptp;

import com.osx.core.context.Context;
import com.osx.core.service.InboundPackage;
import org.ppc.ptp.Pcp;

public class PtpUnaryCallService extends AbstractPtpServiceAdaptor {
    @Override
    protected Pcp.Outbound doService(Context context, InboundPackage<Pcp.Inbound> data) {
        context.setActionType("cancel");
        Pcp.Inbound inbound = data.getBody();
        String  topic  = context.getTopic();
        return null;
    }

}
