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

package com.webank.ai.fate.driver.federation.transfer.communication.consumer;

import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.fate.driver.federation.transfer.manager.TransferMetaHelper;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBrokerListener;
import com.webank.ai.fate.driver.federation.transfer.utils.TransferPojoUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

@Component
@Scope("prototype")
public class TransferBrokerConsumer implements Runnable, TransferBrokerListener {
    private static final Logger LOGGER = LogManager.getLogger();
    private final Object latchLock;
    @Autowired
    private TransferPojoUtils transferPojoUtils;
    @Autowired
    private ErrorUtils errorUtils;
    @Autowired
    private TransferMetaHelper transferMetaHelper;
    private boolean hasProcessed;
    private TransferBroker transferBroker;
    private String transferMetaId;

    private CountDownLatch hasReadyLatch;

    private long startIdleTime;
    private long maxIdleTime = 3600000;
    private int maxConcurrency = 10;
    private boolean isIdleTimeout;
    private boolean inited;

    public TransferBrokerConsumer() {
        this.inited = false;
        this.latchLock = new Object();
        this.hasProcessed = false;
        this.isIdleTimeout = false;
        resetLatch();
    }

    @Override
    public void onListenerChange(TransferBroker brokerToConsume) {
        onReady(brokerToConsume);
    }

    private void onReady(TransferBroker transferBroker) {
        String handler = transferPojoUtils.generateTransferId(transferBroker.getTransferMeta());
        onReady(handler, transferBroker);
    }

    private void onReady(String transferMetaId, TransferBroker transferBroker) {
        if (!inited) {
            this.transferMetaId = transferMetaId;
            this.transferBroker = transferBroker;
            inited = true;
        }

        if (transferBroker.isClosable()) {
            hasReadyLatch.countDown();
            return;
        }

        // LOGGER.info("onListenerChange readyBrokerHandlers.size() = {}, mapSize: {}", readyBrokerHandlers.size(), handlerToBroker.size());
        hasReadyLatch.countDown();
    }

    private void resetLatch() {
        if (hasReadyLatch == null || hasReadyLatch.getCount() == 0L) {
            hasReadyLatch = new CountDownLatch(1);
        }
    }

    @Override
    public void run() {
        LOGGER.info("[FEDERATION][CONSUMER] Consumer starts");
        boolean latchAwaitResult = false;
        Federation.TransferMeta currentTransferMeta;
        startIdleTime = System.currentTimeMillis();

        try {
            /// main loop
            while (!shouldStop()) {
                latchAwaitResult = hasReadyLatch.await(5, TimeUnit.SECONDS);

                long now = System.currentTimeMillis();
                if (!latchAwaitResult) {
                    LOGGER.info("[FEDERATION][CONSUMER] updating status. transferBroker: {}. isClosable: {}, queueSize: {}, isFinished: {}",
                            this, transferBroker.isClosable(), transferBroker.getQueueSize(), transferBroker.isFinished());

                    if (!transferBroker.isReady()) {
                        if (now - startIdleTime > maxIdleTime) {
                            LOGGER.info("[FEDERATION][CONSUMER] timeout for transferMetaId: {}");
                            break;
                        }
                        continue;
                    }
                }

                // LOGGER.info("main loop, after: readyBrokerHandlers.size() = {}, mapSize: {}", readyBrokerHandlers.size(), handlerToBroker.size());
                if (!hasProcessed) {
                    hasProcessed = true;
                }

                // LOGGER.info("[FEDERATION][CONSUMER] processing {}", transferMetaId);

                if (transferBroker == null) {
                    LOGGER.error("[FEDERATION][CONSUMER] transferBroker of transferMetaId {} is null", transferMetaId);
                    throw new IllegalStateException("[FEDERATION][CONSUMER] transferBroker is null: " + transferMetaId);
                }

                currentTransferMeta = transferBroker.getTransferMeta();
                try {
                    if (transferBroker.isReady()) {
/*                        LOGGER.info("[FEDERATION][CONSUMER] running consume action transferMetaId: {}, action type: {}",
                                transferMetaId, transferBroker.getAction().getClass().getSimpleName());*/
                        transferBroker.getAction().onProcess();
                    }
                } catch (Exception e) {
                    LOGGER.error("[FEDERATION][CONSUMER] consumer action error. {}", errorUtils.getStackTrace(e));
                    transferMetaHelper.onError(currentTransferMeta, 313, e);
                    transferBroker.setError(e);
                    throw e;
                }

/*
                LOGGER.info("[FEDERATION][CONSUMER] isReady: {}, isFinished: {}, isClosable: {}",
                        transferBroker.isReady(), transferBroker.isFinished(), transferBroker.isClosable());
*/

                if (!transferBroker.isReady()) {
                    resetLatch();
                }

                startIdleTime = System.currentTimeMillis();
            }
        } catch (Exception e){
            LOGGER.info("[FEDERATION][CONSUMER] error in consumer: {}", errorUtils.getStackTrace(e));
        } finally{
            LOGGER.info("[FEDERATION][CONSUMER] break from consumer: {}", transferMetaId);
            onClose();
        }
    }

    private boolean shouldStop() {
        boolean result = false;
        isIdleTimeout = System.currentTimeMillis() - startIdleTime > maxIdleTime;
        result = hasProcessed && (transferBroker == null || transferBroker.isClosable() || isIdleTimeout);

        return result;
    }

    private void onClose() {
        if (transferBroker.isClosable()) {
            LOGGER.info("[FEDERATION][CONSUMER] ready to close: {}", transferMetaId);
            transferBroker.close();
            LOGGER.info("[FEDERATION][CONSUMER] transferBroker closed: {}", transferMetaId);
        } else {
            LOGGER.warn("[FEDERATION][CONSUMER] trying to close an unclosable transferBroker: {}", transferMetaId);
        }

        hasReadyLatch.countDown();
    }
}
