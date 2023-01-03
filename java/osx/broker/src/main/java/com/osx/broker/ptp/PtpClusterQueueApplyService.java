package com.osx.broker.ptp;

import com.osx.broker.ServiceContainer;
import com.osx.core.context.Context;
import com.osx.core.service.InboundPackage;
import org.ppc.ptp.Osx;


public class PtpClusterQueueApplyService extends AbstractPtpServiceAdaptor {
    @Override
    protected Osx.Outbound doService(Context context, InboundPackage<Osx.Inbound> data) {

        Osx.Inbound inbound
                = data.getBody();
        Osx.Outbound outbound = ServiceContainer.transferQueueManager.applyFromMaster(context, inbound);
        return outbound;
    }

}
