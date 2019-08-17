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

package com.webank.ai.eggroll.framework.roll.service.model;

import com.google.common.collect.Queues;
import com.webank.ai.eggroll.api.storage.Kv;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import javax.annotation.concurrent.GuardedBy;
import java.util.Collection;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

@Component
@Scope("prototype")
// todo: see if this can be merged with TransferBroker and other component
// await seems duplicate
public class OperandBroker {
    private BlockingQueue<Kv.Operand> operandQueue;

    private @GuardedBy("this") volatile boolean isFinished;
    private volatile CountDownLatch readyLatch;
    private final Object readyLatchLock;

    private static final Logger LOGGER = LogManager.getLogger();

    public OperandBroker() {
        this(-1);
    }

    public OperandBroker(int capacity) {
        if (capacity <= 0) {
            this.operandQueue = Queues.newLinkedBlockingQueue();
        } else {
            this.operandQueue = Queues.newLinkedBlockingQueue(capacity);
        }

        this.isFinished = false;
        this.readyLatchLock = new Object();
        resetLatch();
    }

    public void put(Kv.Operand operand) {
        try {
            if (!isFinished() && operand != null) {
                operandQueue.put(operand);
            } else {
                LOGGER.warn("[OPERANDBROKER] null value offered. ignoring");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } finally {
            countDownLatch();
        }
    }

    public boolean addAll(Collection<Kv.Operand> operands) {
        boolean result = false;
        if (operands.size() > 0) {
            result = operandQueue.addAll(operands);
        }

        if (result) {
            countDownLatch();
        }

        return result;
    }

    public Kv.Operand get() {
        Kv.Operand result = null;
        try {
            if (!isClosable()) {
                result = operandQueue.poll(5, TimeUnit.SECONDS);

                //result = operandQueue.take();
            }
        } catch (Exception e) {
            if (result == null) {
                throw new NullPointerException("no element for now");
            }
        }
        resetLatch();

        return result;
    }

    public Kv.Operand peek() {
        return operandQueue.peek();
    }

    public synchronized int drainTo(Collection<Kv.Operand> target) {
        return drainTo(target, Integer.MAX_VALUE);
    }

    public synchronized int drainTo(Collection<Kv.Operand> target, int maxElementSize) {
        int result = operandQueue.drainTo(target, maxElementSize);
        resetLatch();

        return result;
    }

    private void resetLatch() {
        synchronized (readyLatchLock) {
            if ((readyLatch == null || readyLatch.getCount() != 1)
                    && !isFinished()            // "finished" is an expectation of more data. if finished == true, then there is no need to reset latch again
                    && !isReady()) {
                this.readyLatch = new CountDownLatch(1);
            }
        }
    }

    private void countDownLatch() {
        synchronized (readyLatchLock) {
            if (readyLatch != null && readyLatch.getCount() > 0) {
                this.readyLatch.countDown();
            }
        }
    }

    // todo: check thread safety
    public boolean awaitLatch(long timeout, TimeUnit unit) throws InterruptedException {
        if (!operandQueue.isEmpty()) {
            countDownLatch();
        }

        boolean awaitResult = this.readyLatch.await(timeout, unit);

        if (!awaitResult) {
            if (isReady()) {
                countDownLatch();
                awaitResult = true;
            }
        }

        return awaitResult;
    }

    public boolean isReady() {
        return !operandQueue.isEmpty();
    }

    public boolean isFinished() {
        synchronized (this) {
            return isFinished;
        }
    }

    public OperandBroker setFinished() {
        synchronized (this) {
            isFinished = true;
        }
        countDownLatch();
        return this;
    }

    public void close() {
        setFinished();
    }

    public int getQueueSize() {
        return operandQueue.size();
    }

    public boolean isClosable() {
        synchronized (this) {
            return !isReady() && isFinished();
        }
    }
}
