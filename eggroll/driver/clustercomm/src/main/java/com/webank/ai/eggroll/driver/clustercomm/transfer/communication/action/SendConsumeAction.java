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

package com.webank.ai.eggroll.driver.clustercomm.transfer.communication.action;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.driver.clustercomm.transfer.manager.TransferMetaHelper;
import com.webank.ai.eggroll.driver.clustercomm.transfer.api.grpc.client.ProxyClient;
import com.webank.ai.eggroll.driver.clustercomm.transfer.model.TransferBroker;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;


@Component
@Scope("prototype")
public class SendConsumeAction implements TransferQueueConsumeAction {
    @Autowired
    protected TransferMetaHelper transferMetaHelper;
    @Autowired
    private ProxyClient proxyClient;
    @Autowired
    private ToStringUtils toStringUtils;

    private ClusterComm.TransferMeta transferMeta;
    private ClusterComm.TransferStatus currentTransferStatus;

    private TransferBroker transferBroker;
    private BasicMeta.Endpoint target;
    private AtomicInteger pushCount;
    private AtomicBoolean isProxyClientInited;
    private int reinitCount;

    private static final int DEFAULT_REINIT_INTERVAL = 2;
    private final int reinitInterval;
    private static final Logger LOGGER = LogManager.getLogger();

    public SendConsumeAction(TransferBroker transferBroker, BasicMeta.Endpoint target) {
        this.transferBroker = transferBroker;
        this.target = target;
        this.transferMeta = transferBroker.getTransferMeta();
        this.pushCount = new AtomicInteger(0);
        this.isProxyClientInited = new AtomicBoolean(false);
        this.reinitInterval = DEFAULT_REINIT_INTERVAL;
    }

    @Override
    public void onInit() {
        transferMetaHelper.onInit(transferMeta);
        currentTransferStatus = ClusterComm.TransferStatus.INITIALIZING;
    }

    @Override
    public void onProcess() {
        if (currentTransferStatus != ClusterComm.TransferStatus.PROCESSING) {
            transferMetaHelper.onProcess(transferMeta);
            currentTransferStatus = ClusterComm.TransferStatus.PROCESSING;

            boolean pushCountResetResult = pushCount.compareAndSet(0, reinitInterval);
            if (!pushCountResetResult) {
                throw new IllegalStateException("[CLUSTERCOMM] error in reinit pushCount");
            }
        }

        if (!isProxyClientInited.getAndSet(true)) {
            proxyClient.initPush(transferBroker, target);
            LOGGER.info("[CLUSTERCOMM] reiniting push. transferMeta: {}, reinitCount: {}", toStringUtils.toOneLineString(transferMeta), ++reinitCount);
        }

        proxyClient.doPush();

        if (pushCount.decrementAndGet() == 0) {
            if (!isProxyClientInited.getAndSet(false)) {
                throw new IllegalStateException("exception in reinit: should be true when hitting reinitInterval. current value: " + pushCount.get());
            }
            proxyClient.completePush();
        }
    }

    @Override
    public void onComplete() {
        LOGGER.info("[CLUSTERCOMM][SEND][CONSUME][ACTION] trying to complete send action. transferMeta: {}, transferBroker remaining: {}, final reinitCount: {}",
                toStringUtils.toOneLineString(transferMeta), transferBroker.getQueueSize(), reinitCount);
        while (!transferBroker.isClosable()) {
            onProcess();
        }

        if (isProxyClientInited.get()) {
            boolean cleanupResult = isProxyClientInited.compareAndSet(true, false);
            if (!cleanupResult) {
                throw new IllegalStateException("exception in cleanup: fail in onComplete");
            }
            proxyClient.completePush();
        }

        LOGGER.info("[CLUSTERCOMM][SEND][CONSUME][ACTION] actual completes send action. transferMeta: {}, transferBroker remaining: {}, final reinitCount: {}",
                toStringUtils.toOneLineString(transferMeta), transferBroker.getQueueSize(), reinitCount);
        if (!transferBroker.hasError()) {
            transferMetaHelper.onComplete(transferMeta);
            currentTransferStatus = ClusterComm.TransferStatus.COMPLETE;
        } else {
            currentTransferStatus = ClusterComm.TransferStatus.ERROR;
            throw new RuntimeException(transferBroker.getError());
        }
    }
}
