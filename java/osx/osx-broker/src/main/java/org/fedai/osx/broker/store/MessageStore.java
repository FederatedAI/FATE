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
package org.fedai.osx.broker.store;

import org.fedai.osx.broker.message.*;
import org.fedai.osx.broker.queue.*;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.TransferStatus;
import org.fedai.osx.core.exceptions.MappedFileException;
import org.fedai.osx.core.exceptions.TransferQueueInvalidStatusException;
import org.fedai.osx.core.frame.ServiceThread;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class MessageStore {

    protected final AtomicInteger wrotePosition = new AtomicInteger(0);
    Logger logger = LoggerFactory.getLogger(MessageStore.class);
    volatile TransferStatus transferStatus = TransferStatus.INIT;
    long createTimestamp;
    long lastStatusChangeTimestamp;
    long lastWriteTimestamp;
    long lastReadTimestamp;
    TransferQueueManager transferQueueManager;
    MappedFileQueue mappedFileQueue;
    ReentrantLock putMessageLock = new ReentrantLock();
    long beginTimeInLock;
    AppendMessageHandler appendMessageCallback = new DefaultAppendMessageHandler(MetaInfo.MAP_FILE_SIZE);
    AllocateMappedFileService allocateMappedFileService;

    CleanMappedFileThread cleanMappedFileThread = new CleanMappedFileThread();

    public MessageStore(AllocateMappedFileService allocateMappedFileService, String path) {
        allocateMappedFileService = this.allocateMappedFileService;
        mappedFileQueue = new MappedFileQueue(path, MetaInfo.MAP_FILE_SIZE, allocateMappedFileService);
        this.createTimestamp = System.currentTimeMillis();
        this.lastStatusChangeTimestamp = this.createTimestamp;
        this.lastWriteTimestamp = this.createTimestamp;
        cleanMappedFileThread.start();
    }

    public PutMessageResult putMessage(final MessageExtBrokerInner msg) {
        if (transferStatus == TransferStatus.TRANSFERING) {
            long timestamp = System.currentTimeMillis();
            msg.setStoreTimestamp(timestamp);
            if (logger.isTraceEnabled()) {
                logger.trace("put message {}", msg);
            }
            lastWriteTimestamp = timestamp;
            AppendMessageResult result = null;
            String topic = msg.getTopic();
            int queueId = msg.getQueueId();
            long elapsedTimeInLock = 0;
            MappedFile unlockMappedFile = null;
            MappedFile mappedFile = this.mappedFileQueue.getLastMappedFile();

            putMessageLock.lock(); //spin or ReentrantLock ,depending on store config
            try {
                long beginLockTimestamp = System.currentTimeMillis();
                this.beginTimeInLock = beginLockTimestamp;

                // Here settings are stored timestamp, in order to ensure an orderly
                // global
                msg.setStoreTimestamp(beginLockTimestamp);

                if (null == mappedFile || mappedFile.isFull()) {
                    mappedFile = this.mappedFileQueue.getLastMappedFile(0); // Mark: NewFile may be cause noise
                }
                if (null == mappedFile) {
                    logger.error("create mapped file1 error, topic: " + msg.getTopic() );
                    beginTimeInLock = 0;
                    return new PutMessageResult(PutMessageStatus.CREATE_MAPEDFILE_FAILED, null);
                }

                result = mappedFile.appendMessage(msg, this.appendMessageCallback);
                switch (result.getStatus()) {
                    case PUT_OK:
                        break;
                    case END_OF_FILE:
                        if(logger.isTraceEnabled()){
                            logger.trace("");
                        }

                        unlockMappedFile = mappedFile;
                        // Create a new file, re-write the message
                        mappedFile = this.mappedFileQueue.getLastMappedFile(0);
                        if (null == mappedFile) {
                            logger.error("create mapped file error, topic: " + msg.getTopic() );
                            beginTimeInLock = 0;
                            return new PutMessageResult(PutMessageStatus.CREATE_MAPEDFILE_FAILED, result);
                        }
                        result = mappedFile.appendMessage(msg, this.appendMessageCallback);
                        break;
                    case MESSAGE_SIZE_EXCEEDED:
                    case PROPERTIES_SIZE_EXCEEDED:
                        beginTimeInLock = 0;
                        return new PutMessageResult(PutMessageStatus.MESSAGE_ILLEGAL, result);
                    case UNKNOWN_ERROR:
                        beginTimeInLock = 0;
                        return new PutMessageResult(PutMessageStatus.UNKNOWN_ERROR, result);
                    default:
                        beginTimeInLock = 0;
                        return new PutMessageResult(PutMessageStatus.UNKNOWN_ERROR, result);
                }
                elapsedTimeInLock = System.currentTimeMillis() - beginLockTimestamp;
                beginTimeInLock = 0;
            } finally {
                putMessageLock.unlock();
            }
            if (elapsedTimeInLock > 500) {
                logger.warn("[NOTIFYME]putMessage in lock cost time(ms)={}, bodyLength={} AppendMessageResult={}", elapsedTimeInLock, msg.getBody().length, result);
            }
            wrotePosition.addAndGet(result.getWroteBytes());
            PutMessageResult putMessageResult = new PutMessageResult(PutMessageStatus.PUT_OK, result);
            return putMessageResult;
        } else {
            throw new TransferQueueInvalidStatusException("invalid queue status : " + transferStatus);
        }
    }

    public SelectMappedBufferResult consumeOneMessage(long offset) {
        if (transferStatus == TransferStatus.TRANSFERING) {
            Message result = null;
            SelectMappedBufferResult selectMappedBufferResult = this.selectOneMessageByOffset(offset);
            return selectMappedBufferResult;
        } else {
            throw new TransferQueueInvalidStatusException("transfer queue invalid status : " + transferStatus);
        }
    }

    public SelectMappedBufferResult getMessage(final long offset, final int size) {

        if (transferStatus == TransferStatus.TRANSFERING) {
            this.lastReadTimestamp = System.currentTimeMillis();
            if (this.mappedFileQueue != null) {
                MappedFile mappedFile = this.mappedFileQueue.findMappedFileByOffset(offset, offset == 0);
                if (mappedFile != null) {
                    int pos = (int) (offset % MetaInfo.MAP_FILE_SIZE);
                    return mappedFile.selectMappedBuffer(pos, size);
                }
                return null;
            } else {
                throw new MappedFileException();
            }
        } else {
            throw new TransferQueueInvalidStatusException("transfer queue invalid status : " + transferStatus);
        }

    }

    public SelectMappedBufferResult selectOneMessageByOffset(long commitLogOffset) {
        SelectMappedBufferResult sbr = this.getMessage(commitLogOffset, 4);
        if (null != sbr) {
            try {
                // 1 TOTALSIZE
                int size = sbr.getByteBuffer().getInt();
                return this.getMessage(commitLogOffset, size);
            } finally {
                sbr.release();
            }
        }
        return null;
    }

    public long getCreateTimestamp() {
        return createTimestamp;
    }

    public void setCreateTimestamp(long createTimestamp) {
        this.createTimestamp = createTimestamp;
    }

    public TransferStatus getTransferStatus() {
        return transferStatus;
    }

    //    public void setTransferStatus(TransferStatus transferStatus) {
//        this.transferStatus = transferStatus;
//    }
    public AtomicInteger getWrotePosition() {
        return wrotePosition;
    }

    public synchronized void start() {
        if (this.transferStatus == TransferStatus.INIT) {
            this.transferStatus = TransferStatus.TRANSFERING;
        }
    }

    public long getLastReadTimestamp() {
        return lastReadTimestamp;
    }

    public void setLastReadTimestamp(long lastReadTimestamp) {
        this.lastReadTimestamp = lastReadTimestamp;
    }

    public long getLastWriteTimestamp() {
        return lastWriteTimestamp;
    }

    public void setLastWriteTimestamp(long lastWriteTimestamp) {
        this.lastWriteTimestamp = lastWriteTimestamp;
    }

    /**
     * 定期扫描过期文件
     */
    private class CleanMappedFileThread extends ServiceThread {

        @Override
        public String getServiceName() {
            return "CleanMappedFileThread";
        }

        @Override
        public void run() {
            while (true) {
                this.waitForRunning(3600000);
                try {
                    int count = mappedFileQueue.deleteExpiredFileByTime(MetaInfo.PROPERTY_MAPPED_FILE_EXPIRE_TIME, 1000, 1000, false);
                    logger.info("CleanMappedFileThread clean expired mapped file ,count {}", count);
                } catch (Exception e) {
                    logger.error("CleanMappedFileThread clean error", e);
                }
            }
        }
    }
}
