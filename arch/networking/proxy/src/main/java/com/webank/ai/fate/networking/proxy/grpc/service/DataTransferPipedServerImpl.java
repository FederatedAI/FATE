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

package com.webank.ai.fate.networking.proxy.grpc.service;

import com.webank.ai.fate.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.networking.proxy.event.model.PipeHandleNotificationEvent;
import com.webank.ai.fate.networking.proxy.factory.EventFactory;
import com.webank.ai.fate.networking.proxy.factory.GrpcStreamObserverFactory;
import com.webank.ai.fate.networking.proxy.factory.PipeFactory;
import com.webank.ai.fate.networking.proxy.infra.Pipe;
import com.webank.ai.fate.networking.proxy.util.ErrorUtils;
import com.webank.ai.fate.networking.proxy.util.Timeouts;
import com.webank.ai.fate.networking.proxy.util.ToStringUtils;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

@Component
@Scope("prototype")
public class DataTransferPipedServerImpl extends DataTransferServiceGrpc.DataTransferServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger(DataTransferPipedServerImpl.class);
    @Autowired
    private ApplicationEventPublisher applicationEventPublisher;
    @Autowired
    private GrpcStreamObserverFactory grpcStreamObserverFactory;
    @Autowired
    private Timeouts timeouts;
    @Autowired
    private EventFactory eventFactory;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private ErrorUtils errorUtils;
    private Pipe defaultPipe;
    private PipeFactory pipeFactory;

    @Override
    public StreamObserver<Proxy.Packet> push(StreamObserver<Proxy.Metadata> responseObserver) {
        LOGGER.info("[PUSH][SERVER] request received");

        Pipe pipe = getPipe();
        // LOGGER.info("push pipe: {}", pipe);
/*
        PipeHandleNotificationEvent event =
                eventFactory.createPipeHandleNotificationEvent(
                        this, PipeHandleNotificationEvent.Type.PUSH, null, pipe);
        applicationEventPublisher.publishEvent(event);
*/

        StreamObserver<Proxy.Packet> requestObserver = grpcStreamObserverFactory
                .createServerPushRequestStreamObserver(pipe, responseObserver);

        return requestObserver;
    }

    @Override
    public void pull(Proxy.Metadata inputMetadata, StreamObserver<Proxy.Packet> responseObserver) {
        String oneLineStringInputMetadata = toStringUtils.toOneLineString(inputMetadata);
        LOGGER.info("[PULL][SERVER] request received. metadata: {}",
                oneLineStringInputMetadata);

        long overallTimeout = timeouts.getOverallTimeout(inputMetadata);
        long packetIntervalTimeout = timeouts.getPacketIntervalTimeout(inputMetadata);

        Pipe pipe = getPipe();

        LOGGER.info("[PULL][SERVER] pull pipe: {}", pipe);

        PipeHandleNotificationEvent event =
                eventFactory.createPipeHandleNotificationEvent(
                        this, PipeHandleNotificationEvent.Type.PULL, inputMetadata, pipe);
        applicationEventPublisher.publishEvent(event);

        long startTimestamp = System.currentTimeMillis();
        long lastPacketTimestamp = startTimestamp;
        long loopEndTimestamp = lastPacketTimestamp;

        Proxy.Packet packet = null;
        boolean hasReturnedBefore = false;
        int emptyRetryCount = 0;
        Proxy.Packet lastReturnedPacket = null;

        while ((!hasReturnedBefore || !pipe.isDrained())
                && !pipe.hasError()
                && !timeouts.isTimeout(packetIntervalTimeout, lastPacketTimestamp, loopEndTimestamp)
                && !timeouts.isTimeout(overallTimeout, startTimestamp, loopEndTimestamp)) {
            packet = (Proxy.Packet) pipe.read(1, TimeUnit.SECONDS);
            // LOGGER.info("packet is null: {}", Proxy.Packet == null);
            loopEndTimestamp = System.currentTimeMillis();
            if (packet != null) {
                // LOGGER.info("server pull onNext()");
                responseObserver.onNext(packet);
                hasReturnedBefore = true;
                lastReturnedPacket = packet;
                lastPacketTimestamp = loopEndTimestamp;
                emptyRetryCount = 0;
            } else {
                long currentPacketInterval = loopEndTimestamp - lastPacketTimestamp;
                if (++emptyRetryCount % 60 == 0) {
                    LOGGER.info("[PULL][SERVER] pull waiting. current packetInterval: {}, packetIntervalTimeout: {}, metadata: {}",
                            currentPacketInterval, packetIntervalTimeout, oneLineStringInputMetadata);
                }
            }
        }

        boolean hasError = true;
        if (pipe.hasError()) {
            Throwable error = pipe.getError();
            LOGGER.error("[PULL][SERVER] pull finish with error: {}", ExceptionUtils.getStackTrace(error));
            responseObserver.onError(error);

            return;
        }

        StringBuilder sb = new StringBuilder();
        if (timeouts.isTimeout(packetIntervalTimeout, lastPacketTimestamp, loopEndTimestamp)) {
            sb.append("[PULL][SERVER] pull server error: Proxy.Packet interval exceeds timeout: ")
                    .append(packetIntervalTimeout)
                    .append(", metadata: ")
                    .append(oneLineStringInputMetadata)
                    .append(", lastPacketTimestamp: ")
                    .append(lastPacketTimestamp)
                    .append(", loopEndTimestamp: ")
                    .append(loopEndTimestamp);

            String errorMsg = sb.toString();

            LOGGER.error(errorMsg);

            TimeoutException e = new TimeoutException(errorMsg);
            responseObserver.onError(errorUtils.toGrpcRuntimeException(e));
            pipe.onError(e);
        } else if (timeouts.isTimeout(overallTimeout, startTimestamp, loopEndTimestamp)) {
            sb.append("[PULL][SERVER] pull server error: overall process time exceeds timeout: ")
                    .append(overallTimeout)
                    .append(", metadata: ")
                    .append(oneLineStringInputMetadata)
                    .append(", startTimestamp: ")
                    .append(startTimestamp)
                    .append(", loopEndTimestamp: ")
                    .append(loopEndTimestamp);
            String errorMsg = sb.toString();
            LOGGER.error(errorMsg);

            TimeoutException e = new TimeoutException(errorMsg);
            responseObserver.onError(errorUtils.toGrpcRuntimeException(e));
            pipe.onError(e);
        } else {
            responseObserver.onCompleted();
            hasError = false;
            pipe.onComplete();
        }
        LOGGER.info("[PULL][SERVER] server pull finshed. hasReturnedBefore: {}, hasError: {}, metadata: {}",
                hasReturnedBefore, hasError, oneLineStringInputMetadata);
        //LOGGER.warn("pull last returned packet: {}", lastReturnedPacket);
    }

    @Override
    public void unaryCall(Proxy.Packet request, StreamObserver<Proxy.Packet> responseObserver) {
        Proxy.Metadata inputMetadata = request.getHeader();
        String oneLineStringInputMetadata = toStringUtils.toOneLineString(inputMetadata);
        LOGGER.info("[UNARYCALL][SERVER] server unary request received. src: {}, dst: {}",
                toStringUtils.toOneLineString(inputMetadata.getSrc()),
                toStringUtils.toOneLineString(inputMetadata.getDst()));

        long overallTimeout = timeouts.getOverallTimeout(inputMetadata);
        long packetIntervalTimeout = timeouts.getPacketIntervalTimeout(inputMetadata);

        Pipe pipe = getPipe();

        LOGGER.info("[UNARYCALL][SERVER] unary call pipe: {}", pipe);

        PipeHandleNotificationEvent event =
                eventFactory.createPipeHandleNotificationEvent(
                        this, PipeHandleNotificationEvent.Type.UNARY_CALL, request, pipe);
        applicationEventPublisher.publishEvent(event);

        long startTimestamp = System.currentTimeMillis();
        long lastPacketTimestamp = startTimestamp;
        Proxy.Packet packet = null;
        boolean hasReturnedBefore = false;
        int emptyRetryCount = 0;
        long loopEndTimestamp = System.currentTimeMillis();
        while ((!hasReturnedBefore || !pipe.isDrained())
                && !pipe.hasError()
                && !timeouts.isTimeout(overallTimeout, startTimestamp, loopEndTimestamp)) {
            packet = (Proxy.Packet) pipe.read(1, TimeUnit.SECONDS);
            loopEndTimestamp = System.currentTimeMillis();
            if (packet != null) {
                // LOGGER.info("server pull onNext()");
                responseObserver.onNext(packet);
                hasReturnedBefore = true;
                emptyRetryCount = 0;
                break;
            } else {
                long currentOverallWaitTime = loopEndTimestamp - lastPacketTimestamp;

                if (++emptyRetryCount % 60 == 0) {
                    LOGGER.info("[UNARYCALL][SERVER] unary call waiting. current overallWaitTime: {}, packetIntervalTimeout: {}, metadata: {}",
                            currentOverallWaitTime, packetIntervalTimeout, oneLineStringInputMetadata);
                }
            }
        }
        boolean hasError = true;

        if (pipe.hasError()) {
            Throwable error = pipe.getError();
            LOGGER.error("[UNARYCALL][SERVER] unary call finish with error: {}", ExceptionUtils.getStackTrace(error));
            responseObserver.onError(error);

            return;
        }

        if (!hasReturnedBefore) {
            if (timeouts.isTimeout(overallTimeout, startTimestamp, loopEndTimestamp)) {
                String errorMsg = "[UNARYCALL][SERVER] unary call server error: overall process time exceeds timeout: " + overallTimeout
                        + ", metadata: " + oneLineStringInputMetadata
                        + ", lastPacketTimestamp: " + lastPacketTimestamp
                        + ", loopEndTimestamp: " + loopEndTimestamp;
                LOGGER.error(errorMsg);

                TimeoutException e = new TimeoutException(errorMsg);
                responseObserver.onError(errorUtils.toGrpcRuntimeException(e));
                pipe.onError(e);
            } else {
                String errorMsg = "[PULL][SERVER] pull server error: overall process time exceeds timeout: " + overallTimeout
                        + ", metadata: " + oneLineStringInputMetadata
                        + ", startTimestamp: " + startTimestamp
                        + ", loopEndTimestamp: " + loopEndTimestamp;

                TimeoutException e = new TimeoutException(errorMsg);
                responseObserver.onError(errorUtils.toGrpcRuntimeException(e));
                pipe.onError(e);
            }
        } else {
            hasError = false;
            responseObserver.onCompleted();
            pipe.onComplete();
        }

        LOGGER.info("[UNARYCALL][SERVER] server unary call completed. hasReturnedBefore: {}, hasError: {}, metadata: {}",
                hasReturnedBefore, hasError, oneLineStringInputMetadata);
    }

    private void checkNotNull() {
        if (defaultPipe == null && pipeFactory == null) {
            throw new NullPointerException("defaultPipe and pipeFactory are both null");
        }
    }

    private Pipe getPipe() {
        checkNotNull();

        Pipe result = defaultPipe;
        if (pipeFactory != null) {
            result = pipeFactory.create();
        }

        return result;
    }

    public void setDefaultPipe(Pipe defaultPipe) {
        this.defaultPipe = defaultPipe;
    }

    public void setPipeFactory(PipeFactory pipeFactory) {
        this.pipeFactory = pipeFactory;
    }
}
