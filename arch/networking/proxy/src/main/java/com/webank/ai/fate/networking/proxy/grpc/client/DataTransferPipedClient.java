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

package com.webank.ai.fate.networking.proxy.grpc.client;

import com.google.common.base.Preconditions;
import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.networking.proxy.factory.GrpcStreamObserverFactory;
import com.webank.ai.fate.networking.proxy.factory.GrpcStubFactory;
import com.webank.ai.fate.networking.proxy.infra.Pipe;
import com.webank.ai.fate.networking.proxy.infra.ResultCallback;
import com.webank.ai.fate.networking.proxy.infra.impl.PacketQueueSingleResultPipe;
import com.webank.ai.fate.networking.proxy.infra.impl.SingleResultCallback;
import com.webank.ai.fate.networking.proxy.model.ServerConf;
import com.webank.ai.fate.networking.proxy.service.FdnRouter;
import com.webank.ai.fate.networking.proxy.util.ErrorUtils;
import com.webank.ai.fate.networking.proxy.util.ToStringUtils;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

@Component
@Scope("prototype")
public class DataTransferPipedClient {
    private static final Logger LOGGER = LogManager.getLogger(DataTransferPipedClient.class);
    @Autowired
    private GrpcStubFactory grpcStubFactory;
    @Autowired
    private GrpcStreamObserverFactory grpcStreamObserverFactory;
    @Autowired
    private ServerConf serverConf;
    @Autowired
    private FdnRouter fdnRouter;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private ErrorUtils errorUtils;
    private BasicMeta.Endpoint endpoint;
    private boolean needSecureChannel;
    private long MAX_AWAIT_HOURS = 24;

    public DataTransferPipedClient() {
        needSecureChannel = false;
    }

    public void push(Proxy.Metadata metadata, Pipe pipe) {
        String onelineStringMetadata = toStringUtils.toOneLineString(metadata);
        LOGGER.info("[PUSH][CLIENT] client send push to server: {}",
                onelineStringMetadata);
        DataTransferServiceGrpc.DataTransferServiceStub stub = getStub(metadata.getSrc(), metadata.getDst());

        try {
            Proxy.Topic from = metadata.getSrc();
            Proxy.Topic to = metadata.getDst();
            stub = getStub(from, to);
        } catch (Exception e) {
            LOGGER.error("[PUSH][CLIENT] error when creating push stub");
            pipe.onError(e);
        }

        final CountDownLatch finishLatch = new CountDownLatch(1);
        final ResultCallback<Proxy.Metadata> resultCallback = new SingleResultCallback<Proxy.Metadata>();

        StreamObserver<Proxy.Metadata> responseObserver =
                grpcStreamObserverFactory.createClientPushResponseStreamObserver(resultCallback, finishLatch);

        StreamObserver<Proxy.Packet> requestObserver = stub.push(responseObserver);
        LOGGER.info("[PUSH][CLIENT] push stub: {}, metadata: {}",
                stub.getChannel(), onelineStringMetadata);

        int emptyRetryCount = 0;
        Proxy.Packet packet = null;
        do {
            packet = (Proxy.Packet) pipe.read(1, TimeUnit.SECONDS);

            if (packet != null) {
                requestObserver.onNext(packet);
                emptyRetryCount = 0;
            } else {
                ++emptyRetryCount;
                if (emptyRetryCount % 60 == 0) {
                    LOGGER.info("[PUSH][CLIENT] push stub waiting. empty retry count: {}, metadata: {}",
                            emptyRetryCount, onelineStringMetadata);
                }
            }
        } while ((packet != null || !pipe.isDrained()) && emptyRetryCount < 30 && !pipe.hasError());

        LOGGER.info("[PUSH][CLIENT] break out from loop. Proxy.Packet is null? {} ; pipe.isDrained()? {}" +
                        ", pipe.hasError? {}, metadata: {}",
                packet == null, pipe.isDrained(), pipe.hasError(), onelineStringMetadata);

        if (pipe.hasError()) {
            Throwable error = pipe.getError();
            LOGGER.error("[PUSH][CLIENT] push error: {}, metadata: {}",
                    ExceptionUtils.getStackTrace(error), onelineStringMetadata);
            requestObserver.onError(error);

            return;
        }

        requestObserver.onCompleted();
        try {
            finishLatch.await(MAX_AWAIT_HOURS, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            LOGGER.error("[PUSH][CLIENT] client push: finishLatch.await() interrupted");
            requestObserver.onError(errorUtils.toGrpcRuntimeException(e));
            pipe.onError(e);
            Thread.currentThread().interrupt();
            return;
        }

        if (pipe instanceof PacketQueueSingleResultPipe) {
            PacketQueueSingleResultPipe convertedPipe = (PacketQueueSingleResultPipe) pipe;
            if (resultCallback.hasResult()) {
                convertedPipe.setResult(resultCallback.getResult());
            } else {
                LOGGER.warn("No Proxy.Metadata returned in pipe. request metadata: {}",
                        onelineStringMetadata);
            }
        }
        pipe.onComplete();

        LOGGER.info("[PUSH][CLIENT] push closing pipe. metadata: {}",
                onelineStringMetadata);
    }

    public void pull(Proxy.Metadata metadata, Pipe pipe) {
        String onelineStringMetadata = toStringUtils.toOneLineString(metadata);
        LOGGER.info("[PULL][CLIENT] client send pull to server: {}", onelineStringMetadata);
        DataTransferServiceGrpc.DataTransferServiceStub stub = getStub(metadata.getDst(), metadata.getSrc());

        final CountDownLatch finishLatch = new CountDownLatch(1);

        StreamObserver<Proxy.Packet> responseObserver =
                grpcStreamObserverFactory.createClientPullResponseStreamObserver(pipe, finishLatch, metadata);

        stub.pull(metadata, responseObserver);
        LOGGER.info("[PULL][CLIENT] pull stub: {}, metadata: {}",
                stub.getChannel(), onelineStringMetadata);

        try {
            finishLatch.await(MAX_AWAIT_HOURS, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            LOGGER.error("[PULL][CLIENT] client pull: finishLatch.await() interrupted");
            responseObserver.onError(errorUtils.toGrpcRuntimeException(e));
            pipe.onError(e);
            Thread.currentThread().interrupt();
            return;
        }

        responseObserver.onCompleted();
    }

    public void unaryCall(Proxy.Packet packet, Pipe pipe) {
        Preconditions.checkNotNull(packet);
        Proxy.Metadata header = packet.getHeader();
        String onelineStringMetadata = toStringUtils.toOneLineString(header);
        LOGGER.info("[UNARYCALL][CLIENT] client send unary call to server: {}", onelineStringMetadata);
        //LOGGER.info("[UNARYCALL][CLIENT] packet: {}", toStringUtils.toOneLineString(packet));

        DataTransferServiceGrpc.DataTransferServiceStub stub = getStub(
                packet.getHeader().getSrc(), packet.getHeader().getDst());

        final CountDownLatch finishLatch = new CountDownLatch(1);
        StreamObserver<Proxy.Packet> responseObserver = grpcStreamObserverFactory
                .createClientUnaryCallResponseStreamObserver(pipe, finishLatch, packet.getHeader());
        stub.unaryCall(packet, responseObserver);

        LOGGER.info("[UNARYCALL][CLIENT] unary call stub: {}, metadata: {}",
                stub.getChannel(), onelineStringMetadata);

        try {
            finishLatch.await(MAX_AWAIT_HOURS, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            LOGGER.error("[UNARYCALL][CLIENT] client unary call: finishLatch.await() interrupted");
            responseObserver.onError(errorUtils.toGrpcRuntimeException(e));
            pipe.onError(e);
            Thread.currentThread().interrupt();
            return;
        }

        responseObserver.onCompleted();
    }

    private DataTransferServiceGrpc.DataTransferServiceStub getStub(Proxy.Topic from, Proxy.Topic to) {
        if (endpoint == null && !fdnRouter.isAllowed(from, to)) {
            throw new SecurityException("no permission from " + toStringUtils.toOneLineString(from)
                    + " to " + toStringUtils.toOneLineString(to));
        }

        DataTransferServiceGrpc.DataTransferServiceStub stub = null;
        if (endpoint == null) {
            stub = grpcStubFactory.getAsyncStub(to);
        } else {
            stub = grpcStubFactory.getAsyncStub(endpoint);
        }

        LOGGER.info("[ROUTE] route info: {} routed to {}", toStringUtils.toOneLineString(to),
                toStringUtils.toOneLineString(fdnRouter.route(to)));

        fdnRouter.route(from);

        return stub;
    }

    public boolean isNeedSecureChannel() {
        return needSecureChannel;
    }

    public void setNeedSecureChannel(boolean needSecureChannel) {
        this.needSecureChannel = needSecureChannel;
    }

    public BasicMeta.Endpoint getEndpoint() {
        return endpoint;
    }

    public void setEndpoint(BasicMeta.Endpoint endpoint) {
        this.endpoint = endpoint;
    }
}
