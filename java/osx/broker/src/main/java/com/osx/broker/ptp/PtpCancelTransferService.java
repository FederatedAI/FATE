package com.osx.broker.ptp;

import com.osx.broker.ServiceContainer;
import com.osx.core.constant.Dict;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.Context;
import com.osx.core.service.InboundPackage;
import org.ppc.ptp.Osx;


import java.util.List;

public class PtpCancelTransferService extends AbstractPtpServiceAdaptor {

    public PtpCancelTransferService() {
        this.setServiceName("cansel-unary");
    }

    @Override
    protected Osx.Outbound doService(Context context, InboundPackage<Osx.Inbound> data) {

        String sessionId = context.getSessionId();
        String topic = context.getTopic();
        List<String> cleanedTransferId = ServiceContainer.transferQueueManager.cleanByParam(sessionId, topic);
        if (cleanedTransferId != null) {
            for (String transferIdClean : cleanedTransferId) {
                ServiceContainer.consumerManager.onComplete(transferIdClean);
            }
        }
        Osx.Outbound.Builder outBoundBuilder = Osx.Outbound.newBuilder();
        outBoundBuilder.setCode(StatusCode.SUCCESS).setMessage(Dict.SUCCESS);
        return outBoundBuilder.build();
    }


}
