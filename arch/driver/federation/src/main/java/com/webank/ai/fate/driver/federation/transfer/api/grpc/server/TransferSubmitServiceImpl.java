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
import com.webank.ai.fate.api.driver.federation.TransferSubmitServiceGrpc;
import com.webank.ai.eggroll.core.api.grpc.server.GrpcServerWrapper;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.fate.driver.federation.transfer.manager.RecvBrokerManager;
import com.webank.ai.fate.driver.federation.transfer.manager.TransferMetaHelper;
import com.webank.ai.fate.driver.federation.transfer.utils.TransferPojoUtils;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;


@Component
@Scope("prototype")
public class TransferSubmitServiceImpl extends TransferSubmitServiceGrpc.TransferSubmitServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private TransferMetaHelper transferMetaHelper;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private TransferPojoUtils transferPojoUtils;
    @Autowired
    private ErrorUtils errorUtils;
    @Autowired
    private GrpcServerWrapper grpcServerWrapper;
    @Autowired
    private RecvBrokerManager recvBrokerManager;

    @Override
    public void send(Federation.TransferMeta request, StreamObserver<Federation.TransferMeta> responseObserver) {
        LOGGER.info("[FEDERATION][SEND] request received. request: {}, transferMetaId: {}",
                toStringUtils.toOneLineString(request), transferPojoUtils.generateTransferId(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            String transferMetaId = transferPojoUtils.generateTransferId(request);
            Federation.TransferMeta result = transferMetaHelper.get(request);
            if (result == null) {
                LOGGER.info("[FEDERATION][SEND] creating new task for {}", transferMetaId);
                transferMetaHelper.create(request);
                result = request;
            }

            responseObserver.onNext(result);
            responseObserver.onCompleted();
        });
    }

    @Override
    public void recv(Federation.TransferMeta request, StreamObserver<Federation.TransferMeta> responseObserver) {
        LOGGER.info("[FEDERATION][RECV] request received. request: {}, transferMetaId: {}",
                toStringUtils.toOneLineString(request), transferPojoUtils.generateTransferId(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            String transferMetaId = transferPojoUtils.generateTransferId(request);
            Federation.TransferMeta result = recvBrokerManager.getFinishedTask(transferMetaId);
            if (result == null) {
                result = recvBrokerManager.getSubmittedTask(transferMetaId);

                if (result == null) {
                    LOGGER.info("[FEDERATION][RECV] creating new task for {}", transferMetaId);
                    recvBrokerManager.createTask(request);
                    result = request;
                }
            }

            responseObserver.onNext(result);
            responseObserver.onCompleted();
        });
    }

    @Override
    public void checkStatusNow(Federation.TransferMeta request, StreamObserver<Federation.TransferMeta> responseObserver) {
        LOGGER.info("[FEDERATION][CHECKSTATUSNOW] request received. request: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            Federation.TransferMeta result = null;
            Federation.TransferStatus transferStatus = Federation.TransferStatus.UNRECOGNIZED;
            String transferMetaId = transferPojoUtils.generateTransferId(request);
            Federation.TransferType transferType = request.getType();

            switch (transferType) {
                case SEND:
                    result = transferMetaHelper.get(request);
                    break;
                case RECV:
                    result = recvBrokerManager.getFinishedTask(transferMetaId);
                    if (result == null) {
                        result = request;
                    }
                    break;
                default:
                    throw new IllegalArgumentException("Invalid transferType: " + (transferType == null ? transferType : transferType.name()));
            }

            LOGGER.info("[FEDERATION][CHECKSTATUSNOW] result: {}, status: {}", toStringUtils.toOneLineString(result), result.getTransferStatus().name());
            responseObserver.onNext(result);
            responseObserver.onCompleted();
        });
    }

    @Override
    public void checkStatus(Federation.TransferMeta request, StreamObserver<Federation.TransferMeta> responseObserver) {
        LOGGER.info("[FEDERATION][CHECKSTATUS] request received. request: {}, transferMetaId: {}",
                toStringUtils.toOneLineString(request), transferPojoUtils.generateTransferId(request));

        grpcServerWrapper.wrapGrpcServerRunnable(
                responseObserver, () -> {
                    // todo: change to event trigger

                    int iter = 0;

                    Federation.TransferMeta result = null;
                    Federation.TransferStatus transferStatus = Federation.TransferStatus.UNRECOGNIZED;
                    String transferMetaId = transferPojoUtils.generateTransferId(request);
                    Federation.TransferType transferType = request.getType();

                    long startTime = System.currentTimeMillis();
                    CountDownLatch finishLatch = recvBrokerManager.getFinishLatch(transferMetaId);
                    boolean latchWaitResult = false;
                    while (!latchWaitResult) {
                        latchWaitResult = finishLatch.await(10, TimeUnit.SECONDS);

                        switch (transferType) {
                            case SEND:
                                result = transferMetaHelper.get(request);
                                break;
                            case RECV:
                                result = recvBrokerManager.getFinishedTask(transferMetaId);
                                if (result == null) {
                                    result = request;
                                }
                                break;
                            default:
                                throw new IllegalArgumentException("Invalid transferType: " + (transferType == null ? transferType : transferType.name()));
                        }

                        if (result == null) {
                            result = request;
                        }
                        long now = System.currentTimeMillis();
                        long timeInterval = now - startTime;

                        // if not finished (COMPLETE / ERROR / CANCELLED) yet
                        if (!latchWaitResult) {
                            transferStatus = result.getTransferStatus();

                            if (timeInterval >= 300000) {
                                LOGGER.info("[FEDERATION][CHECKSTATUS] breaking. transferMetaId: {}, status: {}, type: {}",
                                        transferMetaId, transferStatus.name(), transferType.name());
                                break;
                            } else {
                                LOGGER.info("[FEDERATION][CHECKSTATUS] waiting. transferMetaId: {}, status: {}, type: {}, latch: {}",
                                        transferMetaId, transferStatus.name(), transferType.name(), finishLatch);
                            }
                        } else {
                            LOGGER.info("[FEDERATION][CHECKSTATUS] finished. transferMetaId: {}, status: {}, type: {}",
                                    transferMetaId, transferStatus.name(), transferType.name());
                        }
                    }

                    responseObserver.onNext(result);
                    responseObserver.onCompleted();
                }
        );
    }
}
