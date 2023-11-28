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
import com.google.common.base.Preconditions;
import com.google.common.cache.*;
import com.google.common.collect.Lists;
import com.google.inject.Inject;
import com.google.inject.Singleton;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.broker.callback.MsgEventCallback;
import org.fedai.osx.broker.consumer.ConsumerManager;
import org.fedai.osx.broker.consumer.EventDriverRule;
import org.fedai.osx.broker.message.AllocateMappedFileService;
import org.fedai.osx.broker.store.MessageStore;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.*;
import org.fedai.osx.core.exceptions.SysException;
import org.fedai.osx.core.frame.ServiceThread;
import org.fedai.osx.core.utils.NetUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.*;
import java.util.concurrent.locks.ReentrantLock;
import static org.fedai.osx.core.config.MetaInfo.*;

@Singleton
public class TransferQueueManager {
    ThreadPoolExecutor errorCallBackExecutor = new ThreadPoolExecutor(1, 2, 1000, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(100));
    Logger logger = LoggerFactory.getLogger(TransferQueueManager.class);
    volatile Map<String, TransferQueueApplyInfo> transferQueueApplyInfoMap = new ConcurrentHashMap<>();
    volatile Set<String> instanceIds = new HashSet<>();
    ConcurrentHashMap<String, AbstractQueue> queueMap = new ConcurrentHashMap<>();
    ConcurrentHashMap<String, Set<String>> sessionQueueMap = new ConcurrentHashMap<>();
    LoadingCache<String, ReentrantLock> transferIdLockMap = CacheBuilder.newBuilder()
            .expireAfterAccess(PROPERTY_MAX_QUEUE_LOCK_LIVE, TimeUnit.SECONDS)
            .concurrencyLevel(4)
            .maximumSize(PROPERTY_MAX_TRANSFER_QUEUE_SIZE)
            .build(new CacheLoader<String, ReentrantLock>() {
                @Override
                public ReentrantLock load(String s) throws Exception {
                    return new ReentrantLock();
                }
            });

    ConcurrentHashMap<EventDriverRule, List<MsgEventCallback>> msgCallBackRuleMap = new ConcurrentHashMap<>();

    @Inject
    ConsumerManager consumerManager;
    MessageStore messageStore;
    AllocateMappedFileService allocateMappedFileService;
    volatile long transferApplyInfoVersion = -1;
    private ServiceThread cleanTask = new ServiceThread() {
        @Override
        public void run() {
            while (true) {
                this.waitForRunning(MetaInfo.PROPERTY_TRANSFER_QUEUE_CHECK_INTERVAL);
                checkAndClean();
            }
        }
        @Override
        public String getServiceName() {
            return "TransferQueueCleanTask";
        }
    };
    public TransferQueueManager() {
        allocateMappedFileService = createAllocateMappedFileService();
        messageStore = createMessageStore(allocateMappedFileService);
        instanceIds.add(MetaInfo.INSTANCE_ID);
        cleanTask.start();
    }

    public static String assembleTopic(String sessionId, String topic) {
        StringBuilder sb = new StringBuilder();
        sb.append(sessionId).append("_").append(topic);
        return sb.toString();
    }

    public MessageStore getMessageStore() {
        return messageStore;
    }

    public void setMessageStore(MessageStore messageStore) {
        this.messageStore = messageStore;
    }

    public MessageStore createMessageStore(
            AllocateMappedFileService allocateMappedFileService) {
        MessageStore messageStore = new MessageStore(allocateMappedFileService
                , MetaInfo.PROPERTY_TRANSFER_FILE_PATH_PRE + File.separator + MetaInfo.INSTANCE_ID + File.separator + "message-store");
        messageStore.start();
        return messageStore;
    }

    AllocateMappedFileService createAllocateMappedFileService() {
        AllocateMappedFileService allocateMappedFileService = new AllocateMappedFileService();
        allocateMappedFileService.start();
        return allocateMappedFileService;
    }

    public List<String> cleanByParam(String sessionId, String topic) {
        logger.info("try to clean {} : {}  session map {}", sessionId, topic, this.sessionQueueMap);
        List<String> result = Lists.newArrayList();
        if (StringUtils.isEmpty(topic)) {
            Set<String> topics = this.sessionQueueMap.get(sessionId);
            if (topics != null) {
                List<String> topicList = Lists.newArrayList(topics);
                for (String tempTopic : topicList) {
                    String indexKey = assembleTopic(sessionId, tempTopic);
                    try {
                        if (this.getQueueByIndexKey(indexKey) != null)
                            destroy(indexKey);
                        result.add(indexKey);
                    } catch (Exception e) {
                        logger.error("destroyInner error {}", indexKey);
                    }
                }
            }
        } else {
            String indexKey = assembleTopic(sessionId, topic);
            try {
                if (queueMap.get(indexKey) != null) {
                    destroy(indexKey);
                    result.add(indexKey);
                }
            } catch (Exception e) {
                logger.error("destroy error {}", indexKey);
            }
        }
        return result;
    }

    private void destroyInner(AbstractQueue queue) {
        queue.destory();
        String sessionId = queue.getSessionId();
        String topic = queue.getTransferId();
        String indexKey = assembleTopic(sessionId, topic);
        queueMap.remove(indexKey);
        Set<String> transferIdSets = this.sessionQueueMap.get(sessionId);
        if (transferIdSets != null) {
            transferIdSets.remove(topic);
            if (transferIdSets.size() == 0) {
                sessionQueueMap.remove(sessionId);
            }
        }
    }

    private void checkAndClean() {
        long now = System.currentTimeMillis();
        logger.info("the total topic size is {}, total session size is {}", queueMap.size(), sessionQueueMap.size());
        queueMap.forEach((transferId, transferQueue) -> {
            try {
                long lastReadTimestamp = transferQueue.getLastReadTimestamp();
                long lastWriteTimestamp = transferQueue.getLastWriteTimestamp();
                long freeTime = now - Math.max(lastReadTimestamp, lastWriteTimestamp);
                if (transferQueue.getTransferStatus() == TransferStatus.ERROR || transferQueue.getTransferStatus() == TransferStatus.FINISH) {
                    destroy(transferId);
                }
                if (freeTime > PROPERTY_QUEUE_MAX_FREE_TIME) {
                    if (logger.isInfoEnabled()) {
                        logger.info("topic : {} freetime  {} need to be destroy", transferId, freeTime);
                    }
                    destroy(transferId);
                }
            } catch (Exception igrone) {
                logger.error("transferQueue clean error ", igrone);
            }
        });
    }
    public ReentrantLock getLock(String transferId) throws ExecutionException {
        return transferIdLockMap.get(transferId);

    }

    public CreateQueueResult createNewQueue(String sessionId, String topic, boolean forceCreateLocal, QueueType queueType) {
        Preconditions.checkArgument(StringUtils.isNotEmpty(topic));
        CreateQueueResult createQueueResult = new CreateQueueResult();
        ReentrantLock transferCreateLock = null;
        try {
            transferCreateLock = getLock(topic);
            transferCreateLock.lock();
            AbstractQueue queue = this.getQueue(sessionId, topic);
            if (queue != null) {
                createQueueResult.setQueue(queue);
                String[] elements = MetaInfo.INSTANCE_ID.split(":");
                createQueueResult.setPort(Integer.parseInt(elements[1]));
                createQueueResult.setRedirectIp(elements[0]);
                return createQueueResult;
            }

                /*
                 * 单机版部署，直接本地建Q
                 */
                createQueueResult.setQueue(localCreate(topic, sessionId, queueType));
                createQueueResult.setPort(MetaInfo.PROPERTY_GRPC_PORT);
                createQueueResult.setRedirectIp(NetUtils.getLocalHost());

            return createQueueResult;
        } catch (Exception e) {
            logger.error("create local queue {} {} error", sessionId, topic, e);
            throw new SysException(StatusCode.PTP_SYSTEM_ERROR, "create queue error");
        } finally {
            if (transferCreateLock != null) {
                transferCreateLock.unlock();
            }
        }
    }
    private void setMsgCallBack(AbstractQueue queue) {
        this.msgCallBackRuleMap.forEach((rule, msgCallbacks) -> {
            if (rule.isMatch(queue)) {
                queue.registerMsgCallback(msgCallbacks);
            } else {
            }
        });
    }

    private AbstractQueue localCreate(String topic, String sessionId, QueueType queueType) {
        logger.info("create local topic {} session {} queue type {}", topic,sessionId, queueType);
        AbstractQueue queue = null;
        switch (queueType) {
            case NORMAL:
                queue = new TransferQueue(topic, this, consumerManager, MetaInfo.PROPERTY_TRANSFER_FILE_PATH_PRE + File.separator + MetaInfo.INSTANCE_ID);
                break;
            case DIRECT:
                queue = new DirectQueue(topic);
                break;
        }
        queue.setSessionId(sessionId);
        queue.start();
        queue.registerDestoryCallback(() -> {
            this.queueMap.remove(assembleTopic(sessionId, topic));
            if (this.sessionQueueMap.get(sessionId) != null) {
                this.sessionQueueMap.get(sessionId).remove(topic);
            }
        });
        setMsgCallBack(queue);
        String indexKey = assembleTopic(sessionId, topic);
        queueMap.put(indexKey, queue);
        if (sessionQueueMap.get(sessionId) == null)
            sessionQueueMap.put(sessionId, new HashSet<>());
        sessionQueueMap.get(sessionId).add(topic);
        return queue;
    }

    public AbstractQueue getQueue(String sessionId, String topic) {
        String indexKey = this.assembleTopic(sessionId, topic);
        return getQueueByIndexKey(indexKey);
    }

    public AbstractQueue getQueueByIndexKey(String indexKey) {
        return queueMap.get(indexKey);
    }

    private void destroy(String indexKey) throws ExecutionException {
        logger.info("start clear topic queue , indexKey = {}", indexKey);

        ReentrantLock transferIdLock = this.getLock(indexKey);
        if (transferIdLock != null) {
            transferIdLock.lock();
        }
        try {
            AbstractQueue transferQueue = getQueueByIndexKey(indexKey);
            if (transferQueue != null) {
                destroyInner(transferQueue);
                //transferIdLockMap.remove(indexKey);
            }
        } finally {
            if (transferIdLock != null) {
                transferIdLock.unlock();
            }
        }
    }

    public void onError(String sessionId, String topic, Throwable throwable) {
        String indexKey = assembleTopic(sessionId, topic);
        AbstractQueue queue = this.getQueueByIndexKey(indexKey);
        if (queue != null) {
            /*
             * 这里需要处理的问题是，当异常发生时，消费者并没有接入，等触发之后才接入
             */
            errorCallBackExecutor.execute(() -> queue.onError(throwable));
        }
        try {
            this.destroy(indexKey);
        } catch (Exception e) {
            logger.error("destory queue {} error", indexKey, e);
        }
    }

    public void onCompleted(String sessionId, String topic) {
        logger.info("transfer queue session {}  topic prepare to destory", sessionId, topic);
        String indexKey = assembleTopic(sessionId, topic);
        AbstractQueue queue = this.getQueueByIndexKey(indexKey);
        if (queue != null) {
            queue.onCompeleted();
        }
        try {
            this.destroy(indexKey);
        } catch (Exception e) {
            logger.error("destory queue {} error", indexKey, e);
        }
    }

    public TransferQueueApplyInfo queryGlobleQueue(String transferId) {
        return this.transferQueueApplyInfoMap.get(transferId);
    }

    public void destroyAll() {
        this.queueMap.forEach((transferId, transferQueue) -> {
            transferQueue.destory();
        });
    }

    public void addMsgCallBackRule(EventDriverRule rule, List<MsgEventCallback> callbacks) {
        this.msgCallBackRuleMap.put(rule, callbacks);
    }


}
