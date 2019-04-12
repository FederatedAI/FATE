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

package com.webank.ai.fate.eggroll.roll.service.model;

import com.google.common.collect.Queues;
import com.webank.ai.fate.api.eggroll.storage.Kv;
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
    private Object latchLock;
    private CountDownLatch readyLatch;

    private static final Logger LOGGER = LogManager.getLogger();

    public OperandBroker() {
        this.operandQueue = Queues.newLinkedBlockingQueue();

        this.isFinished = false;
        this.latchLock = new Object();

        resetLatch();
    }

    public void put(Kv.Operand operand) {
        try {
            if (operand != null) {
                operandQueue.put(operand);
            } else {
                LOGGER.warn("[OPERANDBROKER] null value offered. ignoring");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } finally {
            this.readyLatch.countDown();
        }
    }

    public void addAll(Collection<Kv.Operand> operands) {
        operandQueue.addAll(operands);
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

        if (!isClosable() && operandQueue.isEmpty()) {
            resetLatch();
        }

        return result;
    }

    public Kv.Operand peek() {
        return operandQueue.peek();
    }

    public synchronized int drainTo(Collection<Kv.Operand> target) {
        int result = operandQueue.drainTo(target);
        resetLatch();

        return result;
    }

    public void resetLatch() {
        if (readyLatch == null || readyLatch.getCount() != 1) {
            synchronized (latchLock) {
                this.readyLatch = new CountDownLatch(1);
            }
        }
    }

    public boolean awaitLatch(long timeout, TimeUnit unit) throws InterruptedException {
        return this.readyLatch.await(timeout, unit);
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
        this.readyLatch.countDown();
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
            return !isReady() && isFinished;
        }
    }
}
