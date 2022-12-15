package com.osx.broker.ptp;

import com.osx.core.context.Context;
import com.osx.core.service.InboundPackage;
import com.osx.broker.ServiceContainer;
import org.ppc.ptp.Pcp;

public class PtpClusterQueueApplyService extends AbstractPtpServiceAdaptor {
    @Override
    protected Pcp.Outbound doService(Context context, InboundPackage<Pcp.Inbound> data) {

        Pcp.Inbound inbound
                = data.getBody();
        Pcp.Outbound outbound  = ServiceContainer.transferQueueManager.applyFromMaster(context ,inbound);
        return outbound;
    }

}
