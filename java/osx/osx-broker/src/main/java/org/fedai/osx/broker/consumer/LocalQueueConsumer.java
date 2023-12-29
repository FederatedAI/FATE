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


import lombok.Data;
import org.fedai.osx.broker.message.SelectMappedBufferResult;
import org.fedai.osx.broker.queue.*;
import org.fedai.osx.core.constant.StatusCode;
import org.fedai.osx.core.constant.TransferStatus;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.AckIndexException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicLong;

@Data
public class LocalQueueConsumer implements Consumer<TransferQueueConsumeResult> {

    protected long consumerId;
    Logger logger = LoggerFactory.getLogger(LocalQueueConsumer.class);
    String topic;
    String sessionId;
    AtomicLong consumeOffset = new AtomicLong(1);
    volatile TransferStatus transferStatus = TransferStatus.INIT;
    long createTimestamp = System.currentTimeMillis();
    TransferQueueManager transferQueueManager;

    public LocalQueueConsumer(TransferQueueManager transferQueueManager, long consumerId, String sessionId, String topic) {
        this.consumerId = consumerId;
        this.topic = topic;
        this.sessionId = sessionId;
        this.transferQueueManager = transferQueueManager;
    }

    public boolean checkMsgIsArrive(long consumeOffset) {
        AbstractQueue transferQueue = transferQueueManager.getQueue(sessionId, topic);
        if (transferQueue != null) {
            long indexFileOffset = ((TransferQueue) transferQueue).getIndexQueue().getLogicOffset().get();
            // logger.info("topic {} need consume {} ,  {} inqueue",transferId,consumeOffset, indexFileOffset);
            return consumeOffset <= indexFileOffset;
        }
        return false;
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

    public synchronized TransferQueueConsumeResult consume(OsxContext context, long beginOffset) {
        TransferQueueConsumeResult result;
        long offset = beginOffset;
        TransferQueue transferQueue = (TransferQueue) transferQueueManager.getQueue(sessionId, topic);
        if (transferQueue != null) {
            SelectMappedBufferResult selectMappedBufferResult = null;
            if (offset <= 0) {
                offset = consumeOffset.get();
            }
            result = transferQueue.consumeOneMessage(context, offset);
            //兼容互联互通 ，改成自动ack
            if (StatusCode.SUCCESS.equals(result.getCode())) {
                this.ack(offset);
            }
        } else {
            logger.error("session id {} topic {} is not found", sessionId, topic);
            result = new TransferQueueConsumeResult(StatusCode.TRANSFER_QUEUE_NOT_FIND, null, beginOffset, 0);
        }
        return result;

    }

    @Override
    public void init() {

    }

    @Override
    public void start() {

    }

    @Override
    public void destroy() {

    }
}
