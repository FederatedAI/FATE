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
package org.fedai.osx.broker.service;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import com.google.protobuf.InvalidProtocolBufferException;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.ManagedChannel;
import org.fedai.osx.broker.interceptor.UnaryCallHandleInterceptor;
import org.fedai.osx.broker.router.DefaultFateRouterServiceImpl;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.ActionType;
import org.fedai.osx.core.constant.StatusCode;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.context.Protocol;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.exceptions.NoRouterInfoException;
import org.fedai.osx.core.exceptions.RemoteRpcException;
import org.fedai.osx.core.frame.GrpcConnectionFactory;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.fedai.osx.core.service.InboundPackage;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 用于兼容旧版FATE
 */
@Singleton
public class UnaryCallService extends AbstractServiceAdaptorNew<Proxy.Packet, Proxy.Packet> {

    Logger logger = LoggerFactory.getLogger(UnaryCallService.class);

    @Inject
    DefaultFateRouterServiceImpl routerService;


    public UnaryCallService() {
    }

    @Override
    protected Proxy.Packet doService(OsxContext context, Proxy.Packet req) {
        TransferUtil.assableContextFromProxyPacket(context, req);
        RouterInfo routerInfo = routerService.route(req);
        context.setRouterInfo(routerInfo);
        Proxy.Packet resp = unaryCall(context, req);
        return resp;
    }

//    @Override
//    protected Proxy.Packet doService(OsxContext context, InboundPackage data) {
//        context.setActionType(ActionType.UNARY_CALL.getAlias());
//        Proxy.Packet req = (Proxy.Packet) data.getBody();
//        RouterInfo routerInfo = routerService.route(req);
//        context.setRouterInfo(routerInfo);
//        Proxy.Packet resp = unaryCall(context, req);
//        return resp;
//    }


    protected Proxy.Packet transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {

        throw new RemoteRpcException(exceptionInfo.toString()) ;


    }

    /**
     * 非流式传输
     *
     * @param context
     * @param
     */
    public Proxy.Packet unaryCall(OsxContext context, Proxy.Packet req) {
        Proxy.Packet result = null;
        context.setUri(UriConstants.UNARYCALL);
        context.setActionType(ActionType.UNARY_CALL.name());
        RouterInfo routerInfo=context.getRouterInfo();
        if(routerInfo==null){
            String sourcePartyId = context.getSrcNodeId();
            String desPartyId = context.getDesNodeId();
            throw  new NoRouterInfoException(sourcePartyId+" to "+desPartyId +" found no router info");
        }
        if(routerInfo.getProtocol().equals(Protocol.http)){
            Osx.Inbound  inbound = TransferUtil.
                    buildInboundFromPushingPacket(context,req, MetaInfo.PROPERTY_FATE_TECH_PROVIDER).build();
            Osx.Outbound outbound = (Osx.Outbound)TransferUtil.redirect(context,inbound,routerInfo,true);
            if(outbound!=null) {
                if (outbound.getCode().equals(StatusCode.SUCCESS)) {
                    try {
                        result = Proxy.Packet.parseFrom(outbound.getPayload().toByteArray());
                    } catch (InvalidProtocolBufferException e) {
                        e.printStackTrace();
                    }
                } else {
                    throw new RemoteRpcException(outbound.getMessage());
                }
            }
        }else {
            ManagedChannel managedChannel = null;
            try {
                managedChannel = GrpcConnectionFactory.createManagedChannel(context.getRouterInfo(), false);
                DataTransferServiceGrpc.DataTransferServiceBlockingStub stub = DataTransferServiceGrpc.newBlockingStub(managedChannel);
                result = stub.unaryCall(req);
            } catch (Exception e) {
                logger.error("new channel call exception", e);
            } finally {
                if (managedChannel != null) {
                    managedChannel.shutdown();
                }
            }
        }
        return result;
    }


    @Override
    public Proxy.Packet decode(Object object) {
        return null;
    }

    @Override
    public Osx.Outbound toOutbound(Proxy.Packet response) {
        return null;
    }
}
