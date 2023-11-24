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
package org.fedai.osx.broker.consumer;

import com.google.common.collect.Maps;
import com.google.inject.Inject;
import com.google.inject.Singleton;
import org.fedai.osx.broker.queue.TransferQueueManager;
import org.fedai.osx.core.frame.Lifecycle;
import org.fedai.osx.core.frame.ServiceThread;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

@Singleton
public class ConsumerManager   {
    Logger logger = LoggerFactory.getLogger(ConsumerManager.class);
    @Inject
    TransferQueueManager transferQueueManager;
    @Inject
    ConsumerManager consumerManager;
    ConcurrentHashMap<String, UnaryConsumer> unaryConsumerMap = new ConcurrentHashMap<>();
    AtomicLong consumerIdIndex = new AtomicLong(0);
    ServiceThread monitorThread = new ServiceThread() {
        @Override
        public String getServiceName() {
            return "monitor";
        }

        @Override
        public void run() {
            while (true) {
                try {
                    report();
                } catch (Exception igore) {
                }
                this.waitForRunning(300000);
            }
        }
    };

    ServiceThread longPullingThread = new ServiceThread() {
        @Override
        public String getServiceName() {
            return "longPullingThread";
        }

        @Override
        public void run() {
            int interval = 200;
            final AtomicInteger longPullingWaitingSize = new AtomicInteger(0);
            final AtomicInteger answerCount = new AtomicInteger(0);
            while (true) {
                try {
                    longPullingWaitingSize.set(0);
                    answerCount.set(0);
                    unaryConsumerMap.forEach((transferId, unaryConsumer) -> {
                        try {
                            answerCount.addAndGet(unaryConsumer.answerLongPulling());
                            longPullingWaitingSize.addAndGet(unaryConsumer.getLongPullingQueueSize());
                        } catch (Exception igore) {
                            igore.printStackTrace();
                        }
                    });

                    if (longPullingWaitingSize.get() > 0) {
                        interval = 500;
                    } else {
                        interval = 1000;
                    }
                } catch (Exception igore) {
                }
                this.waitForRunning(interval);
            }
        }
    };
    public ConsumerManager() {
        longPullingThread.start();
        monitorThread.start();
    }
    public void report() {
        AtomicInteger longPullingSize = new AtomicInteger(0);
        longPullingSize.set(0);
        unaryConsumerMap.forEach((transferId, unaryConsumer) -> {
            longPullingSize.addAndGet(unaryConsumer.getLongPullingQueueSize());
        });
        logger.info("consumer monitor,long pulling waiting {} ,total num {}", longPullingSize.get(), unaryConsumerMap.size());
    }
    public UnaryConsumer getOrCreateUnaryConsumer(String sessionId, String topic) {
        String indexKey = TransferQueueManager.assembleTopic(sessionId, topic);
        if (unaryConsumerMap.get(indexKey) == null) {
            UnaryConsumer unaryConsumer =
                    new UnaryConsumer(transferQueueManager, consumerManager, consumerIdIndex.get(), sessionId, topic);
            unaryConsumerMap.putIfAbsent(indexKey, unaryConsumer);
            return unaryConsumerMap.get(indexKey);
        } else {
            return unaryConsumerMap.get(indexKey);
        }
    }
    public void onComplete(String indexKey) {
        if (this.unaryConsumerMap.containsKey(indexKey)) {
            this.unaryConsumerMap.get(indexKey).destroy();
            this.unaryConsumerMap.remove(indexKey);
        }
        logger.info("remove consumer index key {}", indexKey);
    }
}
