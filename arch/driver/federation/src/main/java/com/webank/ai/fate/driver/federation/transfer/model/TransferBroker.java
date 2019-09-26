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

package com.webank.ai.fate.driver.federation.transfer.model;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Queues;
import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.eggroll.core.constant.RuntimeConstants;
import com.webank.ai.fate.driver.federation.transfer.communication.action.TransferQueueConsumeAction;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

@Component
@Scope("prototype")
public class TransferBroker {
    private static final int DEFAULT_QUEUE_CAPACITY = 100_000;
    private Federation.TransferMeta transferMeta;
    private BlockingQueue<ByteString> dataQueue;
    private List<TransferBrokerListener> listeners;
    private int queueCapacity;
    private volatile boolean isFinished;
    private TransferQueueConsumeAction action;
    private BrokerStatus brokerStatus;
    private AtomicInteger producerCount;
    private Throwable error;
    private CountDownLatch closeLatch;

    private final Object isFinishedLock;

    public TransferBroker(Federation.TransferMeta transferMeta) {
        this(transferMeta, DEFAULT_QUEUE_CAPACITY);
    }

    public TransferBroker(Federation.TransferMeta transferMeta, int queueCapacity) {
        this.queueCapacity = queueCapacity;
        this.transferMeta = transferMeta;
        this.dataQueue = Queues.newLinkedBlockingQueue(queueCapacity);
        this.listeners = Lists.newLinkedList();

        this.isFinished = false;
        this.producerCount = new AtomicInteger(1);

        if (transferMeta == null) {
            this.brokerStatus = BrokerStatus.NO_TRANSFER_META;
        } else {
            this.brokerStatus = BrokerStatus.FULL;
        }
        this.isFinishedLock = new Object();
        this.closeLatch = new CountDownLatch(1);
    }

    public TransferBroker(String transferMetaId) {
        this((Federation.TransferMeta) null, DEFAULT_QUEUE_CAPACITY);
    }

    public TransferBroker(String transferMetaId, int queueCapacity) {
        this((Federation.TransferMeta) null, queueCapacity);
    }

    /**
     * Get a copy of current transferMeta
     *
     * @return a new copy of current transferMeta
     */
    public Federation.TransferMeta getTransferMeta() {
        Federation.TransferMeta result = null;
        if (transferMeta != null) {
            result = this.transferMeta.toBuilder().build();
        }
        return result;
    }

    public TransferBroker setTransferMeta(Federation.TransferMeta transferMeta) {
        if (this.transferMeta != null) {
            throw new IllegalStateException("transferMeta has already been set");
        }
        this.transferMeta = transferMeta;
        this.brokerStatus = BrokerStatus.FULL;
        return this;
    }

    public TransferQueueConsumeAction getAction() {
        return action;
    }

    public TransferBroker setAction(TransferQueueConsumeAction action) {
        this.action = action;
        action.onInit();
        return this;
    }

    public void put(byte[] data) {
        put(ByteString.copyFrom(data));
    }

    public void put(ByteString data) {
        if (!isFinished()) {
            try {
                dataQueue.put(data);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException(e);
            }

            notifySubscribers();
        } else {
            throw new IllegalStateException("broker has been marked finished. no more element can be put into here");
        }
    }

    public boolean add(ByteString data) {
        if (!isFinished()) {
            return dataQueue.add(data);
        } else {
            return false;
        }
    }

    public boolean addAll(Collection<ByteString> dataCollection) {
        return dataQueue.addAll(dataCollection);
    }

    public ByteString get() throws Exception {
        ByteString result = dataQueue.poll(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT);

        return result;
    }

    public void notifySubscribers() {
        for (TransferBrokerListener listener : listeners) {
            if (listener != null) {
                listener.onListenerChange(this);
            }
        }
    }

    public synchronized int drainTo(Collection<ByteString> target) {
        return dataQueue.drainTo(target);
    }

    public synchronized int drainTo(Collection<ByteString> target, int maxElementSize) {
        return dataQueue.drainTo(target, maxElementSize);
    }

    public void addSubscriber(TransferBrokerListener listener) {
        Preconditions.checkNotNull(listener, "TransferBrokerListener cannot be null");

        listeners.add(listener);
    }

    public synchronized void setFinished() {
        int curCount = producerCount.decrementAndGet();
        if (curCount == 0) {
            synchronized (isFinishedLock) {
                this.isFinished = true;
            }
        }
        notifySubscribers();
    }

    public synchronized int incrementProducerCount() {
        return this.producerCount.incrementAndGet();
    }

    public Throwable getError() {
        return error;
    }

    public TransferBroker setError(Throwable error) {
        this.error = error;
        this.setFinished();
        action.onComplete();
        return this;
    }

    public boolean isReady() {
        return !dataQueue.isEmpty();
    }

    public boolean isFinished() {
        synchronized (isFinishedLock) {
            return isFinished;
        }
    }

    public boolean isClosable() {
        boolean result = !isReady() && isFinished() && !hasError();

        if (result) {
            closeLatch.countDown();
        }

        return result;
    }

    public boolean awaitClose(long timeout, TimeUnit unit) throws InterruptedException {
        return closeLatch.await(timeout, unit);
    }

    public BrokerStatus getBrokerStatus() {
        return brokerStatus;
    }

    public boolean hasError() {
        return this.error != null;
    }

    public void close() {
        listeners.clear();
        action.onComplete();
    }

    public int getQueueCapacity() {
        return queueCapacity;
    }

    public int getQueueSize() {
        return dataQueue.size();
    }

    public enum BrokerStatus {
        NO_TRANSFER_META,
        FULL;
    }
}
