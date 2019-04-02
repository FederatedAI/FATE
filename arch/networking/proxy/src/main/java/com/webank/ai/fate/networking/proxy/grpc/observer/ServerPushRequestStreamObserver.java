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

package com.webank.ai.fate.networking.proxy.grpc.observer;

import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.networking.proxy.event.model.PipeHandleNotificationEvent;
import com.webank.ai.fate.networking.proxy.factory.EventFactory;
import com.webank.ai.fate.networking.proxy.helper.ModelValidationHelper;
import com.webank.ai.fate.networking.proxy.infra.Pipe;
import com.webank.ai.fate.networking.proxy.infra.impl.PacketQueuePipe;
import com.webank.ai.fate.networking.proxy.manager.StatsManager;
import com.webank.ai.fate.networking.proxy.model.ServerConf;
import com.webank.ai.fate.networking.proxy.model.StreamStat;
import com.webank.ai.fate.networking.proxy.util.ErrorUtils;
import com.webank.ai.fate.networking.proxy.util.PipeUtils;
import com.webank.ai.fate.networking.proxy.util.Timeouts;
import com.webank.ai.fate.networking.proxy.util.ToStringUtils;
import io.grpc.Grpc;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

@Component
@Scope("prototype")
public class ServerPushRequestStreamObserver implements StreamObserver<Proxy.Packet> {
    private static final Logger LOGGER = LogManager.getLogger(ServerPushRequestStreamObserver.class);
    private static final Logger AUDIT = LogManager.getLogger("audit");
    private static final Logger DEBUGGING = LogManager.getLogger("debugging");
    private final StreamObserver<Proxy.Metadata> responseObserver;
    @Autowired
    private ApplicationEventPublisher applicationEventPublisher;
    @Autowired
    private EventFactory eventFactory;
    @Autowired
    private ModelValidationHelper modelValidationHelper;
    @Autowired
    private Timeouts timeouts;
    @Autowired
    private StatsManager statsManager;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private ServerConf serverConf;
    @Autowired
    private ErrorUtils errorUtils;
    @Autowired
    private PipeUtils pipeUtils;
    private Pipe pipe;
    private Proxy.Metadata inputMetadata;
    private StreamStat streamStat;
    private String myCoordinator;
    private long overallStartTimestamp;
    private long overallTimeout;
    private long completionWaitTimeout;
    private String oneLineStringInputMetadata;
    private boolean noError;
    private boolean isAuditEnabled;
    private boolean isDebugEnabled;
    private AtomicLong ackCount;

    public ServerPushRequestStreamObserver(Pipe pipe, StreamObserver<Proxy.Metadata> responseObserver) {
        this.pipe = pipe;
        this.responseObserver = responseObserver;
        this.completionWaitTimeout = Timeouts.DEFAULT_COMPLETION_WAIT_TIMEOUT;
        this.overallTimeout = Timeouts.DEFAULT_OVERALL_TIMEOUT;

        this.noError = true;
        this.ackCount = new AtomicLong(0L);
    }

    @Override
    public void onNext(Proxy.Packet packet) {
        if (inputMetadata == null) {
            overallStartTimestamp = System.currentTimeMillis();
            inputMetadata = packet.getHeader();
            streamStat = new StreamStat(inputMetadata, StreamStat.PUSH);
            oneLineStringInputMetadata = toStringUtils.toOneLineString(inputMetadata);
            statsManager.add(streamStat);

            LOGGER.info(Grpc.TRANSPORT_ATTR_REMOTE_ADDR.toString());

            LOGGER.info("[PUSH][OBSERVER][ONNEXT] metadata: {}", oneLineStringInputMetadata);
            LOGGER.info("[PUSH][OBSERVER][ONNEXT] request src: {}, dst: {}",
                    toStringUtils.toOneLineString(inputMetadata.getSrc()),
                    toStringUtils.toOneLineString(inputMetadata.getDst()));

            if (StringUtils.isBlank(myCoordinator)) {
                myCoordinator = serverConf.getCoordinator();
            }

            if (inputMetadata.hasConf()) {
                overallTimeout = timeouts.getOverallTimeout(inputMetadata);
                completionWaitTimeout = timeouts.getCompletionWaitTimeout(inputMetadata);
            }

            isAuditEnabled = serverConf.isAuditEnabled();
            isDebugEnabled = serverConf.isDebugEnabled();

            // check if topics are valid
            if (!modelValidationHelper.checkTopic(inputMetadata.getDst())
                    || !modelValidationHelper.checkTopic(inputMetadata.getSrc())) {
                onError(new IllegalArgumentException("At least one of topic name, coordinator, role is blank."));
                noError = false;
                return;
            }

            // String operator = inputMetadata.getOperator();

            // LOGGER.info("onNext(): push task name: {}", operator);

            PipeHandleNotificationEvent event =
                    eventFactory.createPipeHandleNotificationEvent(
                            this, PipeHandleNotificationEvent.Type.PUSH, inputMetadata, pipe);
            applicationEventPublisher.publishEvent(event);
        }

        if (noError) {
            pipe.write(packet);
            ackCount.incrementAndGet();
            //LOGGER.info("myCoordinator: {}, Proxy.Packet coordinator: {}", myCoordinator, packet.getHeader().getSrc().getCoordinator());
            if (isAuditEnabled && packet.getHeader().getSrc().getPartyId().equals(myCoordinator)) {
                AUDIT.info(toStringUtils.toOneLineString(packet));
            }

            if (timeouts.isTimeout(overallTimeout, overallStartTimestamp)) {
                onError(new IllegalStateException("push overall wait timeout exceeds overall timeout: " + overallTimeout
                        + ", metadata: " + oneLineStringInputMetadata));
                pipe.close();
                return;
            }

            if (isDebugEnabled) {
                DEBUGGING.info("[PUSH][OBSERVER][ONNEXT] server: {}, ackCount: {}", packet, ackCount.get());
                if (packet.getBody() != null && packet.getBody().getValue() != null) {
                    ByteString value = packet.getBody().getValue();
                    streamStat.increment(value.size());
                    DEBUGGING.info("[PUSH][OBSERVER][ONNEXT] length: {}, metadata: {}",
                            packet.getBody().getValue().size(), oneLineStringInputMetadata);
                } else {
                    DEBUGGING.info("[PUSH][OBSERVER][ONNEXT] length : null, metadata: {}", oneLineStringInputMetadata);
                }
                DEBUGGING.info("-------------");
            }
            //LOGGER.info("push server received size: {}, data size: {}", packet.getSerializedSize(), packet.getBody().getValue().size());
        }
    }

    @Override
    public void onError(Throwable throwable) {
        LOGGER.error("[PUSH][OBSERVER][ONERROR] error in push server: {}, metadata: {}, ackCount: {}",
                Status.fromThrowable(throwable), oneLineStringInputMetadata, ackCount.get());
        LOGGER.error(ExceptionUtils.getStackTrace(throwable));

        pipe.setDrained();

/*        if (Status.fromThrowable(throwable).getCode() != Status.Code.CANCELLED) {
            pipe.onError(throwable);
            responseObserver.onError(errorUtils.toGrpcRuntimeException(throwable));
            streamStat.onError();
        } else {
            noError = false;
            pipe.onComplete();
            LOGGER.info("[PUSH][OBSERVER][ONERROR] connection cancelled. turning into completed.");
            onCompleted();
            streamStat.onComplete();
            return;
        }*/

        pipe.onError(throwable);
        responseObserver.onError(errorUtils.toGrpcRuntimeException(throwable));
        streamStat.onError();
    }

    @Override
    public void onCompleted() {
        long lastestAckCount = ackCount.get();
        LOGGER.info("[PUSH][OBSERVER][ONCOMPLETE] trying to complete task. metadata: {}, ackCount: {}",
                oneLineStringInputMetadata, lastestAckCount);

        long completionWaitStartTimestamp = System.currentTimeMillis();
        long loopEndTimestamp = completionWaitStartTimestamp;
        long waitCount = 0;

        pipe.setDrained();

        /*LOGGER.info("closed: {}, completion timeout: {}, overall timeout: {}",
                pipe.isClosed(),
                timeouts.isTimeout(completionWaitTimeout, completionWaitStartTimestamp, loopEnd),
                timeouts.isTimeout(overallTimeout, overallStartTimestamp, loopEnd));*/
        while (!pipe.isClosed()
                && !timeouts.isTimeout(completionWaitTimeout, completionWaitStartTimestamp, loopEndTimestamp)
                && !timeouts.isTimeout(overallTimeout, overallStartTimestamp, loopEndTimestamp)) {
            // LOGGER.info("waiting for next level result");
            try {
                pipe.awaitClosed(1, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                loopEndTimestamp = System.currentTimeMillis();
            }
            if (++waitCount % 60 == 0) {

                String extraInfo = "";
                if (pipe instanceof PacketQueuePipe) {
                    PacketQueuePipe pqp = (PacketQueuePipe) pipe;
                    extraInfo = "queueSize: " + pqp.getQueueSize();
                }
                LOGGER.info("[PUSH][OBSERVER][ONCOMPLETE] waiting push to complete. wait time: {}. metadata: {}, extrainfo: {}",
                        (loopEndTimestamp - completionWaitStartTimestamp), oneLineStringInputMetadata, extraInfo);
            }
        }

        pipe.onComplete();

        try {
            if (timeouts.isTimeout(completionWaitTimeout, completionWaitStartTimestamp, loopEndTimestamp)) {
                String errmsg = "[PUSH][OBSERVER][ONCOMPLETE] push server completion wait exceeds completionWaitTimeout. "
                        + "completionWaitTimeout: " + completionWaitTimeout
                        + ", metadata: " + oneLineStringInputMetadata
                        + ", completionWaitStartTimestamp: " + completionWaitStartTimestamp
                        + ", loopEndTimestamp: " + loopEndTimestamp
                        + ", ackCount: " + lastestAckCount;
                LOGGER.error(errmsg);
                responseObserver.onError(new TimeoutException(errmsg));
                streamStat.onError();
            } else if (timeouts.isTimeout(overallTimeout, overallStartTimestamp, loopEndTimestamp)) {
                String errmsg = "[PUSH][OBSERVER][ONCOMPLETE] push server overall time exceeds overallTimeout. "
                        + "overallTimeout: " + overallTimeout
                        + ", metadata: " + oneLineStringInputMetadata
                        + ", overallStartTimestamp: " + overallStartTimestamp
                        + ", loopEndTimestamp: " + loopEndTimestamp
                        + ", ackCount: " + lastestAckCount;

                LOGGER.error(errmsg);
                responseObserver.onError(new TimeoutException(errmsg));
                streamStat.onError();
            } else {
                Proxy.Metadata responseMetadata = pipeUtils.getResultFromPipe(pipe);
                if (responseMetadata == null) {
                    LOGGER.warn("[PUSH][OBSERVER][ONCOMPLETE] response Proxy.Metadata is null. inputMetadata: {}",
                            toStringUtils.toOneLineString(responseMetadata));
                }

                responseObserver.onNext(responseMetadata);
                responseObserver.onCompleted();

                LOGGER.info("[PUSH][OBSERVER][ONCOMPLETE] push server complete. inputMetadata: {}",
                        toStringUtils.toOneLineString(responseMetadata));
                streamStat.onComplete();
            }
        } catch (NullPointerException e) {
            LOGGER.error("[PUSH][OBSERVER][ONCOMPLETE] NullPointerException caught in push onComplete. metadata: {}",
                    oneLineStringInputMetadata);
        }
    }
}
