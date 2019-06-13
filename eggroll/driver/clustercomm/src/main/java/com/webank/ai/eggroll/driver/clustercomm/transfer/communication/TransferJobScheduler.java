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

package com.webank.ai.eggroll.driver.clustercomm.transfer.communication;

import com.google.common.base.Preconditions;
import com.google.common.collect.Queues;
import com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.driver.clustercomm.transfer.manager.TransferMetaHelper;
import com.webank.ai.eggroll.driver.clustercomm.transfer.utils.TransferPojoUtils;
import com.webank.ai.eggroll.driver.clustercomm.factory.TransferServiceFactory;
import com.webank.ai.eggroll.driver.clustercomm.transfer.communication.processor.BaseTransferProcessor;
import com.webank.ai.eggroll.driver.clustercomm.transfer.event.TransferJobEvent;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.stereotype.Component;
import org.springframework.util.concurrent.ListenableFuture;
import org.springframework.util.concurrent.ListenableFutureCallback;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

@Component
public class TransferJobScheduler implements Runnable {
    private static final Logger LOGGER = LogManager.getLogger();
    private final Object runningLock;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private TransferServiceFactory transferServiceFactory;
    @Autowired
    private TransferMetaHelper transferMetaHelper;
    @Autowired
    private ThreadPoolTaskExecutor transferJobSchedulerExecutor;
    @Autowired
    private ErrorUtils errorUtils;
    @Autowired
    private TransferPojoUtils transferPojoUtils;

    private BlockingQueue<ClusterComm.TransferMeta> jobQueue;
    // singleton so no explicit synchronization is used for now
    private volatile boolean isRunning;
    private CountDownLatch jobQueueReadyLatch;

    public TransferJobScheduler() {
        this.jobQueue = Queues.newLinkedBlockingQueue(20);
        this.isRunning = false;
        this.runningLock = new Object();
        resetLatch();
    }

    public void submit(ClusterComm.TransferMeta transferMeta) {
        Preconditions.checkNotNull(transferMeta, "transferMeta cannot be null");

        // todo: capacity check and handle
        jobQueue.offer(transferMeta);
        jobQueueReadyLatch.countDown();
    }

    @Override
    public void run() {
        boolean latchWaitResult = false;
        int waitCount = 0;
        while (System.currentTimeMillis() > 0) {
            try {
            while (!latchWaitResult && ++waitCount < 15 && jobQueue.isEmpty()) {
                latchWaitResult = jobQueueReadyLatch.await(1, TimeUnit.SECONDS);
            }

            waitCount = 0;
            latchWaitResult = false;
            resetLatch();

            if (jobQueue.isEmpty()) {
                continue;
            }

            ClusterComm.TransferMeta cur = jobQueue.remove();
            LOGGER.info("[CLUSTERCOMM][SCHEDULER] processing job: {}, executor active: {}, executor max capacity: {}",
                    toStringUtils.toOneLineString(cur), transferJobSchedulerExecutor.getActiveCount(), transferJobSchedulerExecutor.getMaxPoolSize());

            ClusterComm.TransferType type = cur.getType();

            BaseTransferProcessor processor = null;
            switch (type) {
                case SEND:
                    processor = transferServiceFactory.createSendProcessor(cur);
                    break;
                case RECV:
                    processor = transferServiceFactory.createRecvProcessor(cur);
                    break;
                default:
                    transferMetaHelper.onError(cur, 200, "Invalid transfer type: " + type);
                    break;
            }

            String transferMetaId = transferPojoUtils.generateTransferId(cur);
            if (processor != null) {
                LOGGER.info("[CLUSTERCOMM][SCHEDULER] ready to submit job. transferMetaId: {}, type: {}, processorType: {}, ",
                        transferMetaId, type.name(), processor.getClass().getSimpleName());

                CountDownLatch countDownLatch = new CountDownLatch(1);
                ListenableFuture<?> listenableFuture = transferJobSchedulerExecutor.submitListenable(processor);
                listenableFuture.addCallback(new ListenableFutureCallback<Object>() {
                    @Override
                    public void onFailure(Throwable throwable) {
                        LOGGER.error("[CLUSTERCOMM][SCHEDULER] processor failed: transferMetaId: {}, exception: {}",
                                transferMetaId, errorUtils.getStackTrace(throwable));
                        countDownLatch.countDown();
                    }

                    @Override
                    public void onSuccess(Object o) {
                        LOGGER.info("[CLUSTERCOMM][SCHEDULER] processor success. transferMetaId: {}", transferMetaId);
                        countDownLatch.countDown();
                    }
                });

/*                    boolean awaitResult = false;
                while (!awaitResult) {
                    boolean isDone = listenableFuture.isDone();
                    boolean isCancelled = listenableFuture.isCancelled();
                    LOGGER.info("[CLUSTERCOMM][SCHEDULER] transferMetaId: {}, isDone: {}, isCancelled: {}",
                            transferMetaId, isDone, isCancelled);

                    if (isDone || isCancelled) {
                        LOGGER.info("[CLUSTERCOMM][SCHEDULER] transferMetaId: {}, breaking", transferMetaId);
                    }
                    awaitResult = countDownLatch.await(10, TimeUnit.SECONDS);
                }*/
            } else {
                LOGGER.error("[CLUSTERCOMM][SCHEDULER][FATAL] processor is null. transferMetaId: {}. type: {}",
                        transferMetaId, type.name());
            }

            } catch (Exception e) {
                LOGGER.error(errorUtils.getStackTrace(e));
            }
        }
    }

    @Async
    @EventListener
    public void handleTransferJobEvent(TransferJobEvent transferJobEvent) {
        Preconditions.checkNotNull(transferJobEvent);

        submit(transferJobEvent.getTransferMeta());
    }

    private void resetLatch() {
        if (jobQueueReadyLatch == null || jobQueueReadyLatch.getCount() == 0) {
            jobQueueReadyLatch = new CountDownLatch(1);
        }
    }
}
