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
package com.osx.broker.consumer;


import com.osx.api.context.Context;
import com.osx.broker.ServiceContainer;
import com.osx.broker.message.SelectMappedBufferResult;
import com.osx.broker.queue.Consumer;
import com.osx.broker.queue.TransferQueue;
import com.osx.core.constant.StatusCode;
import com.osx.core.constant.TransferStatus;
import com.osx.core.exceptions.AckIndexException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicLong;

public class LocalQueueConsumer implements Consumer<TransferQueue.TransferQueueConsumeResult> {

    Logger logger = LoggerFactory.getLogger(LocalQueueConsumer.class);
    protected long consumerId;
    String transferId;
    AtomicLong consumeOffset = new AtomicLong(1);
    volatile TransferStatus transferStatus = TransferStatus.INIT;
    long createTimestamp = System.currentTimeMillis();

    public LocalQueueConsumer(long consumerId, String transferId) {
        this.consumerId = consumerId;
        this.transferId = transferId;
    }

    public long getConsumerId() {
        return consumerId;
    }

    public void setConsumerId(long consumerId) {
        this.consumerId = consumerId;
    }

    public String getTransferId() {
        return transferId;
    }

    public void setTransferId(String transferId) {
        this.transferId = transferId;
    }

    public boolean checkMsgIsArrive(long consumeOffset) {
        TransferQueue transferQueue = ServiceContainer.transferQueueManager.getQueue(transferId);
        if (transferQueue != null) {
            long indexFileOffset = transferQueue.getIndexQueue().getLogicOffset().get();
            return consumeOffset <= indexFileOffset;
        }
        return false;
    }

    public TransferStatus getTransferStatus() {
        return transferStatus;
    }

    public void setTransferStatus(TransferStatus transferStatus) {
        this.transferStatus = transferStatus;
    }

    public long getCreateTimestamp() {
        return createTimestamp;
    }

    public void setCreateTimestamp(long createTimestamp) {
        this.createTimestamp = createTimestamp;
    }

    public long addConsumeCount(int size) {
        return this.consumeOffset.addAndGet(size);
    }

    public long ack(long index) {
        long currentIndex = this.consumeOffset.get();
        if (index != currentIndex) {
            throw new AckIndexException("ack invalid index ,current : " + currentIndex + " ack : " + index);
        } else {
            return this.consumeOffset.addAndGet(1);
        }
    }

    public long getConsumeOffset() {
        return this.consumeOffset.get();
    }

    public void setConsumeOffset(AtomicLong consumeOffset) {
        this.consumeOffset = consumeOffset;
    }

    public synchronized TransferQueue.TransferQueueConsumeResult consume(Context context, long beginOffset) {
        TransferQueue.TransferQueueConsumeResult result;
        long offset = beginOffset;
        TransferQueue transferQueue = ServiceContainer.transferQueueManager.getQueue(transferId);
        if (transferQueue != null) {
            SelectMappedBufferResult selectMappedBufferResult = null;
            if (offset <= 0) {
                offset = consumeOffset.get();
            }
            result = transferQueue.consumeOneMessage(context, offset);

        } else {
            logger.error("transfer Id {} is not found", transferId);
            result = new TransferQueue.TransferQueueConsumeResult(StatusCode.TRANSFER_QUEUE_NOT_FIND, null, beginOffset, 0);
        }
        return result;

    }

}
