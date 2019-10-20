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

package com.webank.ai.fate.driver.federation.transfer.api.grpc.server;

import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.fate.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.eggroll.core.api.grpc.server.GrpcServerWrapper;
import com.webank.ai.eggroll.core.api.grpc.server.GrpcServerWrapper;
import com.webank.ai.eggroll.core.constant.StringConstants;
import com.webank.ai.eggroll.core.constant.StringConstants;
import com.webank.ai.fate.driver.federation.factory.TransferServiceFactory;
import com.webank.ai.fate.driver.federation.transfer.api.grpc.observer.PushServerRequestStreamObserver;
import com.webank.ai.fate.driver.federation.transfer.manager.RecvBrokerManager;
import com.webank.ai.fate.driver.federation.transfer.utils.TransferProtoMessageUtils;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicBoolean;

@Component
@Scope("prototype")
public class ProxyServiceImpl extends DataTransferServiceGrpc.DataTransferServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private TransferServiceFactory transferServiceFactory;
    @Autowired
    private GrpcServerWrapper grpcServerWrapper;
    @Autowired
    private TransferProtoMessageUtils transferProtoMessageUtils;
    @Autowired
    private RecvBrokerManager recvBrokerManager;

    @Override
    public PushServerRequestStreamObserver push(StreamObserver<Proxy.Metadata> responseObserver) {
        LOGGER.info("[FEDERATION][PROXY][PUSH] request received");

        // todo: implement this in framework
        final ServerCallStreamObserver<Proxy.Metadata> serverCallStreamObserver
                = (ServerCallStreamObserver<Proxy.Metadata>) responseObserver;
        serverCallStreamObserver.disableAutoInboundFlowControl();

        final AtomicBoolean wasReady = new AtomicBoolean(false);

        serverCallStreamObserver.setOnReadyHandler(() -> {
            if (serverCallStreamObserver.isReady() && wasReady.compareAndSet(false, true)) {
                serverCallStreamObserver.request(1);
            }
        });

        return transferServiceFactory.createPushServerRequestStreamObserver(responseObserver, wasReady);
    }

    @Override
    public void unaryCall(Proxy.Packet request, StreamObserver<Proxy.Packet> responseObserver) {
        LOGGER.info("[FEDERATION][PROXY][UNARY] request received: {}", request.getHeader().getTask().getTaskId());
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            Proxy.Metadata header = request.getHeader();
            Proxy.Command command = header.getCommand();
            String commandName = command.getName();

            Proxy.Task task = header.getTask();
            if (StringConstants.SEND_START.equals(commandName)) {
                LOGGER.info("[FEDERATION][PROXY][UNARY] mark start: {}", task.getTaskId());
                Federation.TransferMeta transferMeta = transferProtoMessageUtils.extractTransferMetaFromPacket(request);

                recvBrokerManager.markStart(transferMeta);

                responseObserver.onNext(request);
                responseObserver.onCompleted();
            } else if (StringConstants.SEND_END.equals(commandName)) {
                LOGGER.info("[FEDERATION][PROXY][UNARY] mark end: {}", task.getTaskId());
                Federation.TransferMeta transferMeta = transferProtoMessageUtils.extractTransferMetaFromPacket(request);

                recvBrokerManager.markEnd(transferMeta);

                responseObserver.onNext(request);
                responseObserver.onCompleted();
            } else {
                throw new UnsupportedOperationException("command " + commandName + " not supported");
            }
        });
    }
}
