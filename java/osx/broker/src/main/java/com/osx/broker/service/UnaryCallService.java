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
package com.osx.broker.service;
import com.osx.core.constant.ActionType;
import com.osx.core.context.Context;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.service.AbstractServiceAdaptor;
import com.osx.core.service.InboundPackage;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.Deadline;
import io.grpc.ManagedChannel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 用于兼容旧版FATE
 */
public class UnaryCallService extends AbstractServiceAdaptor<Proxy.Packet, Proxy.Packet> {

    Logger logger = LoggerFactory.getLogger(UnaryCallService.class);


    public UnaryCallService() {

    }

    @Override
    protected Proxy.Packet doService(Context context, InboundPackage data) {
        context.setActionType(ActionType.UNARY_CALL.getAlias());
        Proxy.Packet req = (Proxy.Packet) data.getBody();
        Proxy.Packet resp = unaryCall(context, req);
        //logger.info("uncary req {} resp {}", req, resp);
        return resp;
    }


    protected Proxy.Packet transformExceptionInfo(Context context, ExceptionInfo exceptionInfo) {
        return null;
    }

    /**
     * 非流式传输
     *
     * @param context
     * @param
     */
    public Proxy.Packet unaryCall(Context context, Proxy.Packet req) {
        Deadline endDeadline = null;
        boolean isPolling = false;

        ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(context.getRouterInfo(),true);
        DataTransferServiceGrpc.DataTransferServiceBlockingStub stub = DataTransferServiceGrpc.newBlockingStub(managedChannel);
        Proxy.Packet result = null;
        result = stub.unaryCall(req);
        return result;
    }


}
