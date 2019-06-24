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

package com.webank.ai.eggroll.driver.clustercomm.transfer.manager;

import com.google.common.collect.Maps;
import com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.driver.clustercomm.factory.TransferServiceFactory;
import com.webank.ai.eggroll.driver.clustercomm.transfer.event.TransferJobEvent;
import com.webank.ai.eggroll.driver.clustercomm.transfer.model.TransferBroker;
import com.webank.ai.eggroll.driver.clustercomm.transfer.utils.TransferPojoUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

@Component
public class RecvBrokerManager {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private TransferServiceFactory transferServiceFactory;
    @Autowired
    private TransferPojoUtils transferPojoUtils;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private ApplicationEventPublisher applicationEventPublisher;

    private Object holderLock;
    private Object finishLatchLock;
    private Map<String, TransferBroker> transferMetaIdToBrokerHolder;
    private Map<String, ClusterComm.TransferMeta> transferMetaIdToPassedInTransferMeta;
    private Map<String, CountDownLatch> transferMetaIdToPassedInTransferMetaArriveLatches;
    private Map<String, ClusterComm.TransferMeta> finishedTransferMetas;
    private volatile Map<String, CountDownLatch> transferMetaIdToFinishLatches;
    private Map<String, ClusterComm.TransferMeta> createdTasks;

    public RecvBrokerManager() {
        this.transferMetaIdToBrokerHolder = Maps.newConcurrentMap();
        this.transferMetaIdToPassedInTransferMeta = Maps.newConcurrentMap();
        this.transferMetaIdToPassedInTransferMetaArriveLatches = Maps.newConcurrentMap();
        this.holderLock = new Object();
        this.finishLatchLock = new Object();
        this.finishedTransferMetas = Maps.newConcurrentMap();
        this.transferMetaIdToFinishLatches = Maps.newConcurrentMap();
        this.createdTasks = Maps.newConcurrentMap();
    }

    public void createTask(ClusterComm.TransferMeta transferMeta) {
        String transferMetaId = transferPojoUtils.generateTransferId(transferMeta);

        if (getSubmittedTask(transferMetaId) != null) {
            return;
        }

        createdTasks.putIfAbsent(transferMetaId, transferMeta);

        applicationEventPublisher.publishEvent(new TransferJobEvent(this, transferMeta));
    }

    public void createRecvTaskFromPassedInTransferMetaId(String transferMetaId) {
        ClusterComm.TransferMeta passedInTransferMeta = transferMetaIdToPassedInTransferMeta.get(transferMetaId);
        ClusterComm.TransferMeta transferMeta = passedInTransferMeta.toBuilder().setType(ClusterComm.TransferType.RECV).build();

        if (transferMeta != null) {
            LOGGER.info("[RECV][MANAGER] createTask: got transferMeta from transferMetaIdToPassedInTransferMeta: {}", transferMetaId);
            createTask(transferMeta);
        } else {
            LOGGER.info("[RECV][MANAGER] createTask: transferMeta: {} not exists", transferMetaId);
        }
    }

    /**
     * @param transferMeta
     * @return true if successfully created, false if already exists
     */
    public boolean markStart(ClusterComm.TransferMeta transferMeta) {
        boolean result = false;
        String transferMetaId = transferPojoUtils.generateTransferId(transferMeta);
        if (transferMetaIdToPassedInTransferMeta.containsKey(transferMetaId)) {
            return result;
        }

        result = true;
        transferMetaIdToPassedInTransferMeta.putIfAbsent(transferMetaId, transferMeta);
        createIfNotExistsInternal(transferMetaId, transferMeta);

        CountDownLatch arriveLatch = transferMetaIdToPassedInTransferMetaArriveLatches.get(transferMetaId);

        if (arriveLatch != null) {
            arriveLatch.countDown();
        }

        return result;
    }

    public boolean markEnd(ClusterComm.TransferMeta transferMeta) {
        boolean result = false;
        String transferMetaId = transferPojoUtils.generateTransferId(transferMeta);

        ClusterComm.TransferMeta passedInTransferMeta = transferMetaIdToPassedInTransferMeta.get(transferMetaId);
        if (passedInTransferMeta == null) {
            return result;
        }

        result = true;

        TransferBroker transferBroker = transferMetaIdToBrokerHolder.get(transferMetaId);
        if (transferBroker == null) {
            throw new IllegalStateException("transferBroker not exists in holder");
        }

        transferBroker.setFinished();

        return result;
    }

    private synchronized TransferBroker createIfNotExistsInternal(String transferMetaId, ClusterComm.TransferMeta transferMeta) {
        LOGGER.info("[RECV][MANAGER] createIfNotExistsInternal: {}, {}", transferMetaId, toStringUtils.toOneLineString(transferMeta));
        TransferBroker result = null;

        if (StringUtils.isBlank(transferMetaId)) {
            throw new IllegalArgumentException("key " + transferMetaId + " is blank");
        }

        if (transferMeta != null) {
            transferMetaId = transferPojoUtils.generateTransferId(transferMeta);
        }

        if (!transferMetaIdToBrokerHolder.containsKey(transferMetaId)) {
            synchronized (holderLock) {
                if (!transferMetaIdToBrokerHolder.containsKey(transferMetaId)) {
                    LOGGER.info("[RECV][MANAGER] creating for: {}, {}", transferMetaId, toStringUtils.toOneLineString(transferMeta));
                    result = transferServiceFactory.createTransferBroker(transferMetaId, 1000);
                    transferMetaIdToBrokerHolder.putIfAbsent(transferMetaId, result);
                }
            }
        }
        result = transferMetaIdToBrokerHolder.get(transferMetaId);

        return result;
    }

    public CountDownLatch getFinishLatch(String transferMetaId) {
        boolean newlyCreated = false;
        if (!transferMetaIdToFinishLatches.containsKey(transferMetaId)) {
            synchronized (finishLatchLock) {
                if (!transferMetaIdToFinishLatches.containsKey(transferMetaId)) {
                    transferMetaIdToFinishLatches.putIfAbsent(transferMetaId, new CountDownLatch(1));
                    newlyCreated = true;
                }
            }
        }

        CountDownLatch result = transferMetaIdToFinishLatches.get(transferMetaId);
        LOGGER.info("[RECV][MANAGER] getting finish latch. transferMetaId: {}, finishLatch: {}, newlyCreated: {}",
                transferMetaId, result, newlyCreated);

        return result;
    }

    public TransferBroker createIfNotExists(String transferMetaId) {
        return createIfNotExistsInternal(transferMetaId, null);
    }

    public TransferBroker createIfNotExists(ClusterComm.TransferMeta transferMeta) {
        String transferMetaId = transferPojoUtils.generateTransferId(transferMeta);

        return createIfNotExistsInternal(transferMetaId, transferMeta);
    }

    public TransferBroker remove(String transferMetaId) {
        LOGGER.info("[RECV][MANAGER] removing: {}", transferMetaId);
        TransferBroker result = null;
        if (StringUtils.isBlank(transferMetaId)) {
            LOGGER.warn("putting blank transferMetaId to map");
        } else {
            result = transferMetaIdToBrokerHolder.remove(transferMetaId);
            transferMetaIdToPassedInTransferMeta.remove(transferMetaId);
        }

        return result;
    }

    public TransferBroker getBroker(String transferMetaId) {
        TransferBroker result = transferMetaIdToBrokerHolder.get(transferMetaId);
        LOGGER.info("[RECV][MANAGER] getting broker. transferMetaId: {}, result: {}", transferMetaId, result);
        return result;
    }

    public ClusterComm.TransferMeta getFinishedTask(String transferMetaId) {
        return finishedTransferMetas.get(transferMetaId);
    }

    public ClusterComm.TransferMeta getSubmittedTask(String transferMetaId) {
        return createdTasks.get(transferMetaId);
    }

    public boolean setFinishedTask(ClusterComm.TransferMeta transferMeta, ClusterComm.TransferStatus transferStatus) {
        String transferMetaId = transferPojoUtils.generateTransferId(transferMeta);
        boolean result = false;

        if (finishedTransferMetas.containsKey(transferMetaId)) {
            LOGGER.warn("[RECV][MANAGER] finished task has already exist: {}", transferMetaId);
        } else {
            LOGGER.info("[RECV][MANAGER] updating status to {} for transferMetaId: {}",
                    transferStatus, transferMetaId);
            ClusterComm.TransferMeta transferMetaWithStatusChanged
                    = transferMeta.toBuilder().setTransferStatus(transferStatus).build();
            finishedTransferMetas.put(transferMetaId, transferMetaWithStatusChanged);

            result = true;
        }

        if (transferStatus.equals(ClusterComm.TransferStatus.COMPLETE)
                || transferStatus.equals(ClusterComm.TransferStatus.ERROR)
                || transferStatus.equals(ClusterComm.TransferStatus.CANCELLED)) {
            CountDownLatch finishLatch = getFinishLatch(transferMetaId);

            LOGGER.info("[RECV][MANAGER] counting down latch: {}, transeferMetaId: {}", finishLatch, transferMetaId);
            finishLatch.countDown();
        }

        return result;
    }

    public ClusterComm.TransferMeta blockingGetPassedInTransferMeta(String transferMetaId,
                                                                   long timeout,
                                                                   TimeUnit timeunit)
            throws InterruptedException {
        boolean latchWaitResult = false;
        if (!transferMetaIdToPassedInTransferMeta.containsKey(transferMetaId)) {
            CountDownLatch arriveLatch = new CountDownLatch(1);
            transferMetaIdToPassedInTransferMetaArriveLatches.putIfAbsent(transferMetaId, arriveLatch);

            latchWaitResult = arriveLatch.await(timeout, timeunit);
        }

        return transferMetaIdToPassedInTransferMeta.get(transferMetaId);
    }

    public ClusterComm.TransferMeta getPassedInTransferMetaNow(String transferMetaId) {
        return transferMetaIdToPassedInTransferMeta.get(transferMetaId);
    }

    public Set<Map.Entry<String, TransferBroker>> getEntrySet() {
        return transferMetaIdToBrokerHolder.entrySet();
    }
}
