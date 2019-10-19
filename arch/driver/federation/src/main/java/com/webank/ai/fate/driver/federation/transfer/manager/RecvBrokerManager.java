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

package com.webank.ai.fate.driver.federation.transfer.manager;

import com.google.common.collect.Maps;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.fate.driver.federation.factory.TransferServiceFactory;
import com.webank.ai.fate.driver.federation.transfer.event.TransferJobEvent;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import com.webank.ai.fate.driver.federation.transfer.utils.TransferPojoUtils;
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
    private Map<String, Federation.TransferMeta> transferMetaIdToPassedInTransferMeta;
    private Map<String, CountDownLatch> transferMetaIdToPassedInTransferMetaArriveLatches;
    private Map<String, Federation.TransferMeta> finishedTransferMetas;
    private volatile Map<String, CountDownLatch> transferMetaIdToFinishLatches;
    private Map<String, Federation.TransferMeta> createdTasks;

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

    public void createTask(Federation.TransferMeta transferMeta) {
        String transferMetaId = transferPojoUtils.generateTransferId(transferMeta);

        if (getSubmittedTask(transferMetaId) != null) {
            return;
        }

        createdTasks.putIfAbsent(transferMetaId, transferMeta);

        applicationEventPublisher.publishEvent(new TransferJobEvent(this, transferMeta));
    }

    public void createRecvTaskFromPassedInTransferMetaId(String transferMetaId) {
        Federation.TransferMeta passedInTransferMeta = transferMetaIdToPassedInTransferMeta.get(transferMetaId);
        Federation.TransferMeta transferMeta = passedInTransferMeta.toBuilder().setType(Federation.TransferType.RECV).build();

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
    public boolean markStart(Federation.TransferMeta transferMeta) {
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

    public boolean markEnd(Federation.TransferMeta transferMeta) {
        boolean result = false;
        String transferMetaId = transferPojoUtils.generateTransferId(transferMeta);

        Federation.TransferMeta passedInTransferMeta = transferMetaIdToPassedInTransferMeta.get(transferMetaId);
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

    private synchronized TransferBroker createIfNotExistsInternal(String transferMetaId, Federation.TransferMeta transferMeta) {
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

    public TransferBroker createIfNotExists(Federation.TransferMeta transferMeta) {
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

    public Federation.TransferMeta getFinishedTask(String transferMetaId) {
        return finishedTransferMetas.get(transferMetaId);
    }

    public Federation.TransferMeta getSubmittedTask(String transferMetaId) {
        return createdTasks.get(transferMetaId);
    }

    public boolean setFinishedTask(Federation.TransferMeta transferMeta, Federation.TransferStatus transferStatus) {
        String transferMetaId = transferPojoUtils.generateTransferId(transferMeta);
        boolean result = false;

        if (finishedTransferMetas.containsKey(transferMetaId)) {
            LOGGER.warn("[RECV][MANAGER] finished task has already exist: {}", transferMetaId);
        } else {
            LOGGER.info("[RECV][MANAGER] updating status to {} for transferMetaId: {}",
                    transferStatus, transferMetaId);
            Federation.TransferMeta transferMetaWithStatusChanged
                    = transferMeta.toBuilder().setTransferStatus(transferStatus).build();
            finishedTransferMetas.put(transferMetaId, transferMetaWithStatusChanged);

            result = true;
        }

        if (transferStatus.equals(Federation.TransferStatus.COMPLETE)
                || transferStatus.equals(Federation.TransferStatus.ERROR)
                || transferStatus.equals(Federation.TransferStatus.CANCELLED)) {
            CountDownLatch finishLatch = getFinishLatch(transferMetaId);

            LOGGER.info("[RECV][MANAGER] counting down latch: {}, transeferMetaId: {}", finishLatch, transferMetaId);
            finishLatch.countDown();
        }

        return result;
    }

    public Federation.TransferMeta blockingGetPassedInTransferMeta(String transferMetaId,
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

    public Federation.TransferMeta getPassedInTransferMetaNow(String transferMetaId) {
        return transferMetaIdToPassedInTransferMeta.get(transferMetaId);
    }

    public Set<Map.Entry<String, TransferBroker>> getEntrySet() {
        return transferMetaIdToBrokerHolder.entrySet();
    }
}
