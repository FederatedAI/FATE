package com.osx.broker.ptp;

import com.osx.core.constant.ActionType;
import com.osx.core.context.Context;
import com.osx.core.exceptions.RemoteRpcException;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.router.RouterInfo;
import com.osx.core.service.InboundPackage;
import io.grpc.ManagedChannel;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class PtpUnaryCallService extends AbstractPtpServiceAdaptor {

    Logger logger = LoggerFactory.getLogger(PtpUnaryCallService.class);
    @Override
    protected Osx.Outbound doService(Context context, InboundPackage<Osx.Inbound> data) {
        context.setActionType(ActionType.UNARY_CALL_NEW.getAlias());
        RouterInfo routerInfo = context.getRouterInfo();
        Osx.Inbound inbound = data.getBody();
        String host = routerInfo.getHost();
        Integer port = routerInfo.getPort();
        ManagedChannel managedChannel=GrpcConnectionFactory.createManagedChannel(routerInfo,true);
        PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub blockingStub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
        Osx.Outbound outbound= null;
        try {
             outbound = blockingStub.invoke(inbound);
        }catch(io.grpc.StatusRuntimeException e){
            logger.error("remote rpc error ：router info {}",routerInfo);
            throw  new RemoteRpcException("remote rpc error");
        }
        return outbound;
    }

}
