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
package org.fedai.osx.broker.queue;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import lombok.Data;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.broker.callback.CompleteCallback;
import org.fedai.osx.broker.callback.DestoryCallback;
import org.fedai.osx.broker.callback.ErrorCallback;
import org.fedai.osx.broker.callback.MsgEventCallback;
import org.fedai.osx.broker.constants.MessageFlag;
import org.fedai.osx.broker.consumer.ConsumerManager;
import org.fedai.osx.broker.message.MessageDecoder;
import org.fedai.osx.broker.message.MessageExt;
import org.fedai.osx.broker.message.MessageExtBrokerInner;
import org.fedai.osx.broker.message.SelectMappedBufferResult;
import org.fedai.osx.broker.store.IndexQueue;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.constant.StatusCode;
import org.fedai.osx.core.constant.TransferStatus;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.PutMessageException;
import org.fedai.osx.core.exceptions.TransferQueueInvalidStatusException;
import org.fedai.osx.core.queue.TranferQueueInfo;
import org.fedai.osx.core.service.OutboundPackage;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;
@Data
public class TransferQueue extends AbstractQueue{
    Logger logger = LoggerFactory.getLogger(TransferQueue.class);
    AtomicReferenceArray<String> receivedMsgIds = new AtomicReferenceArray<>(MetaInfo.PROPERTY_TRANSFER_CACHED_MSGID_SIZE);
    private Cache<String, OutboundPackage<Osx.Outbound>> receivedMsgCache;
    IndexQueue indexQueue;
    boolean hasEventMsgDestoryCallback = false;

    public TransferQueue(String transferId, TransferQueueManager transferQueueManager, ConsumerManager consumerManager,String path) {
        this.transferId = transferId;
        this.transferQueueManager = transferQueueManager;
        this.createTimestamp = System.currentTimeMillis();
        this.lastStatusChangeTimestamp = this.createTimestamp;
        this.lastWriteTimestamp = this.createTimestamp;
        this.indexQueue = new IndexQueue(transferId, path, MetaInfo.PROPERTY_INDEX_MAP_FILE_SIZE);
        this.consumerManager = consumerManager;
        initReceivedMsgCache();
    }

    public synchronized boolean checkMsgIdDuplicate(String msgId) {
        for (int i = 0; i < receivedMsgIds.length(); i++) {
            String tempMsgId = receivedMsgIds.get(i);
            if (msgId.equals(tempMsgId)) {
                return true;
            }
        }
        return false;
    }

    private synchronized PutMessageResult putMessage(final MessageExtBrokerInner msg)  {

        if (transferStatus == TransferStatus.TRANSFERING) {
            String msgId = msg.getMsgId();
            this.lastWriteTimestamp = System.currentTimeMillis();
            PutMessageResult putMessageResult = transferQueueManager.messageStore.putMessage(msg);
            if (putMessageResult.isOk()) {

                int cacheIdx = wrotePosition.addAndGet(1) % MetaInfo.PROPERTY_TRANSFER_CACHED_MSGID_SIZE;
                receivedMsgIds.set(cacheIdx, msgId);
                long beginWriteOffset = putMessageResult.getAppendMessageResult().getWroteOffset();
                int size = putMessageResult.getAppendMessageResult().getWroteBytes();
                putMessageResult.setMsgLogicOffset(indexQueue.putMessagePositionInfoWrapper(beginWriteOffset, size));
                //todo 这里需要修改，用另外的队列类型来做，就不再需要持久化
                if (this.msgCallbacks.size() > 0) {
                    try {
                        for (MsgEventCallback msgCallback : this.msgCallbacks) {
                            msgCallback.callback(consumerManager,this, msg);
                        }
                    }catch(Exception  e){
                        e.printStackTrace();
                        logger.error("topic {} callback error",msg.getTopic(),e);
                        throw new PutMessageException("topic " + msg.getTopic() + " callback error");
                    }
                }
            } else {
                logger.info("topic {} put msg error {}",transferId,putMessageResult.getPutMessageStatus());
                throw new PutMessageException("topic " + msg.getTopic() + " put message error");
            }
            return putMessageResult;
        } else {
            logger.error("topic {} is not ready",transferId);
            throw new TransferQueueInvalidStatusException("invalid queue status : " + transferStatus);
        }
    }
    @Override
    public synchronized void putMessage(OsxContext context, Object data , MessageFlag  messageFlag,String msgCode)  {
        context.putData(Dict.MESSAGE_FLAG, messageFlag.name());
        MessageExtBrokerInner messageExtBrokerInner = MessageDecoder.buildMessageExtBrokerInner(context.getTopic(), (byte[])data, msgCode, messageFlag,
                context.getSrcNodeId(),
                context.getDesNodeId());
        messageExtBrokerInner.getProperties().put(Dict.SESSION_ID, sessionId);
        messageExtBrokerInner.getProperties().put(Dict.SOURCE_COMPONENT, context.getSrcComponent() != null ? context.getSrcComponent() : "");
        messageExtBrokerInner.getProperties().put(Dict.DES_COMPONENT, context.getDesComponent() != null ? context.getDesComponent() : "");
        PutMessageResult putMessageResult= this.putMessage(messageExtBrokerInner);
        if (putMessageResult.getPutMessageStatus() != PutMessageStatus.PUT_OK) {
            throw new PutMessageException("put status " + putMessageResult.getPutMessageStatus());
        }
        long logicOffset = putMessageResult.getMsgLogicOffset();
        context.putData(Dict.CURRENT_INDEX, this.getIndexQueue().getLogicOffset().get());
    }
    @Override
    public TransferQueueConsumeResult consumeOneMessage(OsxContext context, long requestIndex) {
        TransferQueueConsumeResult transferQueueConsumeResult;

        if (transferStatus == TransferStatus.TRANSFERING) {
            this.lastReadTimestamp = System.currentTimeMillis();
            long logicIndex = indexQueue.getLogicOffset().get();

            context.putData(Dict.REQUEST_INDEX, requestIndex);
            //context.setCurrentMsgIndex(logicIndex);
            context.putData(Dict.CURRENT_INDEX, logicIndex);
            if (requestIndex <= logicIndex) {
                SelectMappedBufferResult indexBufferResult = this.indexQueue.getIndexBuffer(requestIndex);
                if (indexBufferResult != null) {
                    long pyOffset = indexBufferResult.getByteBuffer().getLong();
                    SelectMappedBufferResult msgBufferResult = this.transferQueueManager.getMessageStore().consumeOneMessage(pyOffset);
                    transferQueueConsumeResult = new TransferQueueConsumeResult(StatusCode.SUCCESS, msgBufferResult, requestIndex, logicIndex);
                    MessageExt message = MessageDecoder.decode(transferQueueConsumeResult.getSelectMappedBufferResult().getByteBuffer());
                    transferQueueConsumeResult.setMessage(message);
                } else {
                    transferQueueConsumeResult = new TransferQueueConsumeResult(StatusCode.INVALID_INDEXFILE_DETAIL, null, requestIndex, logicIndex);
                }
            } else {
                transferQueueConsumeResult = new TransferQueueConsumeResult(StatusCode.CONSUME_NO_MESSAGE, null, requestIndex, logicIndex);
            }
        } else {
            throw new TransferQueueInvalidStatusException("transfer queue invalid status : " + transferStatus);
        }
        return transferQueueConsumeResult;
    }

    public synchronized void destory() {

        this.indexQueue.destroy();
        super.destory();
    }

    public synchronized void start() {
        logger.info("topic {} start ", transferId);
        if (this.transferStatus == TransferStatus.INIT) {
            this.transferStatus = TransferStatus.TRANSFERING;
        }
    }



    public void cacheReceivedMsg(String msgId, OutboundPackage<Osx.Outbound> outboundPackage) {

        if(StringUtils.isNotEmpty(msgId))
            receivedMsgCache.put(msgId, outboundPackage);
    }

    public OutboundPackage<Osx.Outbound> getReceivedMsgCache(String sessionId) {

        return receivedMsgCache.getIfPresent(sessionId);
    }

    private void initReceivedMsgCache() {
        if (receivedMsgCache == null) {
            CacheBuilder<Object, Object> cacheBuilder = CacheBuilder.newBuilder().maximumSize(MetaInfo.PRODUCE_MSG_CACHE_MAX_SIZE);
            if (MetaInfo.PRODUCE_MSG_CACHE_TIMEOUT != null && MetaInfo.PRODUCE_MSG_CACHE_TIMEOUT > 0) {
                cacheBuilder.expireAfterWrite(MetaInfo.PRODUCE_MSG_CACHE_TIMEOUT, TimeUnit.MILLISECONDS);
            }
            receivedMsgCache = cacheBuilder.build();
        }
    }

    public TranferQueueInfo getTransferQueueInfo() {
        TranferQueueInfo transferQueueInfo = new TranferQueueInfo();
        transferQueueInfo.setTransferId(transferId);
        transferQueueInfo.setCreateTimestamp(createTimestamp);
        transferQueueInfo.setLastReadTimestamp(lastReadTimestamp);
        transferQueueInfo.setLastWriteTimestamp(lastWriteTimestamp);
        transferQueueInfo.setTransferStatus(transferStatus);
        transferQueueInfo.setLogicOffset(indexQueue.getLogicOffset().get());
        return transferQueueInfo;
    }




}
