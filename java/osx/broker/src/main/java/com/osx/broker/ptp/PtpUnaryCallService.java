/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
