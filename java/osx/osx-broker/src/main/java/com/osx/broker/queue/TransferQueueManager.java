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
package com.osx.broker.queue;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.osx.api.constants.Protocol;
import com.osx.api.router.RouterInfo;
import com.osx.broker.ServiceContainer;
import com.osx.broker.callback.MsgEventCallback;
import com.osx.broker.consumer.EventDriverRule;
import com.osx.broker.message.AllocateMappedFileService;
import com.osx.broker.store.MessageStore;
import com.osx.core.config.MasterInfo;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.DeployMode;
import com.osx.core.constant.Dict;
import com.osx.core.constant.StatusCode;
import com.osx.core.constant.TransferStatus;
import com.osx.core.exceptions.CreateTopicErrorException;
import com.osx.core.exceptions.RemoteRpcException;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.frame.ServiceThread;
import com.osx.core.ptp.TargetMethod;
import com.osx.core.utils.JsonUtil;
import com.osx.core.utils.NetUtils;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import org.apache.commons.lang3.StringUtils;
import org.apache.zookeeper.KeeperException;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

public class TransferQueueManager {
    final String ZK_QUEUE_PREFIX = "/FATE-TRANSFER/QUEUE";
    final String MASTER_PATH = "/FATE-TRANSFER/MASTER";
    final String ZK_COMPONENTS_PREFIX = "/FATE-COMPONENTS/osx";
    ThreadPoolExecutor errorCallBackExecutor = new ThreadPoolExecutor(1, 2, 1000, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(100));
    ThreadPoolExecutor completeCallBackExecutor = new ThreadPoolExecutor(1, 2, 1000, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(100));
    ThreadPoolExecutor destroyCallBackExecutor = new ThreadPoolExecutor(1, 2, 1000, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(100));
    Logger logger = LoggerFactory.getLogger(TransferQueueManager.class);
    volatile Map<String, TransferQueueApplyInfo> transferQueueApplyInfoMap = new ConcurrentHashMap<>();
    volatile Map<String, TransferQueueApplyInfo> masterQueueApplyInfoMap = new ConcurrentHashMap<>();
    Map<String, Integer> clusterTransferQueueCountMap = Maps.newHashMap();
    volatile Set<String> instanceIds = new HashSet<>();
    ConcurrentHashMap<String, TransferQueue> transferQueueMap = new ConcurrentHashMap<>();
    ConcurrentHashMap<String, Set<String>> sessionQueueMap = new ConcurrentHashMap<>();
    ConcurrentHashMap<String, ReentrantLock> transferIdLockMap = new ConcurrentHashMap<>();
    ConcurrentHashMap<EventDriverRule, List<MsgEventCallback>> msgCallBackRuleMap = new ConcurrentHashMap<>();

    public MessageStore getMessageStore() {
        return messageStore;
    }

    public void setMessageStore(MessageStore messageStore) {
        this.messageStore = messageStore;
    }

    MessageStore messageStore;
    AllocateMappedFileService allocateMappedFileService;
    volatile long transferApplyInfoVersion = -1;

    public MessageStore createMessageStore(
            AllocateMappedFileService allocateMappedFileService) {
        MessageStore messageStore = new MessageStore(allocateMappedFileService
                , MetaInfo.PROPERTY_TRANSFER_FILE_PATH_PRE + File.separator + MetaInfo.INSTANCE_ID + File.separator + "message-store");
        messageStore.start();
        return messageStore;

    }

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


    AllocateMappedFileService createAllocateMappedFileService() {
        AllocateMappedFileService allocateMappedFileService = new AllocateMappedFileService();
        allocateMappedFileService.start();
        return allocateMappedFileService;
    }

    public TransferQueueManager() {
        allocateMappedFileService = createAllocateMappedFileService();
        messageStore = createMessageStore(allocateMappedFileService);
        instanceIds.add(MetaInfo.INSTANCE_ID);
        if (MetaInfo.isCluster()) {
            boolean pathExists = ServiceContainer.zkClient.checkExists(ZK_QUEUE_PREFIX);
            if (!pathExists) {
                ServiceContainer.zkClient.create(ZK_QUEUE_PREFIX, false);
            }
            List<String> initApplyInfo = ServiceContainer.zkClient.addChildListener(ZK_QUEUE_PREFIX, (path, children) -> parseApplyInfo(children));
            parseApplyInfo(initApplyInfo);
            ServiceContainer.zkClient.create(ZK_COMPONENTS_PREFIX + "/" + MetaInfo.INSTANCE_ID, true);
            List<String> initInstanceIds = ServiceContainer.zkClient.addChildListener(ZK_COMPONENTS_PREFIX, (path, children) -> handleClusterInstanceId(children));
            ServiceContainer.zkClient.addDataListener(MASTER_PATH, (path, data, type) -> {
                logger.info("master event {} {}", type, data);
                if (data != null) {
                    try {
                        MetaInfo.masterInfo = parseMasterInfo((String) data);
                    } catch (Exception e) {
                        logger.info("parse master info error", e);
                    }
                } else {
                    electMaster();
                }
            });
            handleClusterInstanceId(initInstanceIds);
        }
        cleanTask.start();
    }

    public Set<String> getInstanceIds() {
        return instanceIds;
    }

    public void setInstanceIds(Set<String> instanceIds) {
        this.instanceIds = instanceIds;
    }

    public Map<String, TransferQueueApplyInfo> getGlobalTransferQueueMap() {
        return transferQueueApplyInfoMap;
    }

    public boolean isMaster() {
        return MetaInfo.INSTANCE_ID.equals(MetaInfo.masterInfo.getInstanceId());
    }

    private void electMaster() {
        try {
            MasterInfo electMasterInfo = new MasterInfo();
            electMasterInfo.setInstanceId(MetaInfo.INSTANCE_ID);
            logger.info("try to elect master !!!");
            ServiceContainer.zkClient.createEphemeral(MASTER_PATH, JsonUtil.object2Json(electMasterInfo));
            logger.info("this instance is master !!!");
            this.masterQueueApplyInfoMap = this.transferQueueApplyInfoMap;
            new ServiceThread() {
                @Override
                public void run() {
                    while (true) {
                        doMasterWork();
                        waitForRunning(1000);
                    }
                }

                @Override
                public String getServiceName() {
                    return "MasterWorkThread";
                }
            }.start();
        } catch (Exception e) {
            if (e instanceof KeeperException.NodeExistsException) {
                logger.info("master is already elected");
            }
        }
    }

    /**
     * 平衡的策略暂时没有开发
     *
     */
    private String doClusterBalance(String transferId,
                                    String instanceId,
                                    String sessionId) {
        return instanceId;
    }

    private void doMasterWork() {
        long current = System.currentTimeMillis();
        //统计各个实例中队列数
        Map<String, Integer> temp = Maps.newHashMap();
        transferQueueApplyInfoMap.forEach((k, v) -> {
            String instanceId = v.getInstanceId();
            if (instanceIds.contains(instanceId)) {
                temp.merge(instanceId, 1, Integer::sum);
                ;
            }
        });
        this.clusterTransferQueueCountMap = temp;
        masterQueueApplyInfoMap.forEach((k, v) -> {
            if (current - v.getApplyTimestamp() > 10000) {
                if (transferQueueApplyInfoMap.get(k) == null) {
                    masterQueueApplyInfoMap.remove(k);
                }
            }
            ;
        });
    }

    private MasterInfo parseMasterInfo(String masterContent) {
        return JsonUtil.json2Object(masterContent, MasterInfo.class);
    }

    private void handleClusterInstanceId(List<String> children) {
        this.instanceIds.clear();
        this.instanceIds.addAll(children);
        if (logger.isInfoEnabled()) {
            logger.info("instance change : {}", instanceIds);
        }
    }

    private synchronized void parseApplyInfo(List<String> children) {
        Set<String> childSet = Sets.newHashSet(children);
        Set<String> intersectionSet = Sets.intersection(transferQueueApplyInfoMap.keySet(), childSet);
        Set<String> needAddSet;
        needAddSet = Sets.difference(childSet, intersectionSet);
        Set<String> needRemoveSet = Sets.difference(transferQueueApplyInfoMap.keySet(), intersectionSet);
        if (logger.isInfoEnabled()) {
            logger.info("cluster apply info add {} remove {}", needAddSet, needRemoveSet);
        }
        needRemoveSet.forEach(k -> transferQueueApplyInfoMap.remove(k));
        needAddSet.forEach(k -> {
            try {
                String content = ServiceContainer.zkClient.getContent(buildZkPath(k));
                TransferQueueApplyInfo transferQueueApplyInfo = JsonUtil.json2Object(content, TransferQueueApplyInfo.class);
                if (transferQueueApplyInfo != null) {
                    transferQueueApplyInfoMap.put(k, transferQueueApplyInfo);
                }
            } catch (Exception e) {
                logger.error("parse apply info from zk error", e);
            }
        });
    }

    ;

    public List<String> cleanByParam(String sessionId, String paramTransferId) {
        List<String> result = Lists.newArrayList();
        if (StringUtils.isEmpty(paramTransferId)) {
            Set<String> transferIdSets = this.sessionQueueMap.get(sessionId);
            if (transferIdSets != null) {
                List<String> transferIdList = Lists.newArrayList(transferIdSets);
                for (String transferId : transferIdList) {
                    try {
                        if (transferQueueMap.get(transferId) != null)
                            destroy(transferId);
                        result.add(transferId);
                    } catch (Exception e) {
                        logger.error("destroyInner error {}", transferId);
                    }
                }
            }
        } else {
            try {
                if (transferQueueMap.get(paramTransferId) != null) {
                    destroy(paramTransferId);
                    result.add(paramTransferId);
                }
            } catch (Exception e) {
                logger.error("destroy error {}", paramTransferId);
            }
        }
        return result;
    }

    private void destroyInner(TransferQueue transferQueue) {
        transferQueue.destory();
        transferQueueMap.remove(transferQueue.getTransferId());
        String sessionId = transferQueue.getSessionId();
        Set<String> transferIdSets = this.sessionQueueMap.get(sessionId);
        if (transferIdSets != null) {
            transferIdSets.remove(transferQueue.getTransferId());
            if (transferIdSets.size() == 0) {
                sessionQueueMap.remove(sessionId);
            }
        }
    }

    private void checkAndClean() {
        long now = System.currentTimeMillis();
        logger.info("the total topic size is {}, total session size is {}", transferQueueMap.size(), sessionQueueMap.size());
        transferQueueMap.forEach((transferId, transferQueue) -> {
            try {
                long lastReadTimestamp = transferQueue.getLastReadTimestamp();
                long lastWriteTimestamp = transferQueue.getLastWriteTimestamp();
                long freeTime = now - Math.max(lastReadTimestamp, lastWriteTimestamp);
                if (transferQueue.getTransferStatus() == TransferStatus.ERROR || transferQueue.getTransferStatus() == TransferStatus.FINISH) {
                    destroy(transferId);
                }
                if (freeTime > MetaInfo.PROPERTY_QUEUE_MAX_FREE_TIME) {
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


    public Enumeration<String> getAllTransferIds() {
        return transferQueueMap.keys();
    }

    public List<TransferQueue> getTransferQueues(List<String> transferIds) {
        List<TransferQueue> result = Lists.newArrayList();
        for (String transferId : transferIds) {
            result.add(this.transferQueueMap.get(transferId));
        }
        return result;
    }


    public synchronized TransferQueueApplyInfo handleClusterApply(String transferId,
                                                                  String instanceId,
                                                                  String sessionId) {

        TransferQueueApplyInfo transferQueueApplyInfo = this.masterQueueApplyInfoMap.get(transferId);
        if (transferQueueApplyInfo != null) {
            return transferQueueApplyInfo;
        } else {
            long current = System.currentTimeMillis();
            TransferQueueApplyInfo newTransferQueueApplyInfo = new TransferQueueApplyInfo();
            doClusterBalance(transferId, instanceId, sessionId);
            newTransferQueueApplyInfo.setTransferId(transferId);
            newTransferQueueApplyInfo.setInstanceId(instanceId);
            newTransferQueueApplyInfo.setSessionId(sessionId);
            newTransferQueueApplyInfo.setApplyTimestamp(current);
            this.masterQueueApplyInfoMap.put(transferId, newTransferQueueApplyInfo);
            return newTransferQueueApplyInfo;
        }
    }


    public ReentrantLock getLock(String  transferId){
        ReentrantLock transferCreateLock = transferIdLockMap.get(transferId);
        if (transferCreateLock == null) {
            transferIdLockMap.putIfAbsent(transferId, new ReentrantLock(false));
        }
        transferCreateLock = transferIdLockMap.get(transferId);
        return  transferCreateLock;
    }




    public CreateQueueResult createNewQueue(String transferId, String sessionId, boolean forceCreateLocal) {
        Preconditions.checkArgument(StringUtils.isNotEmpty(transferId));
        CreateQueueResult createQueueResult = new CreateQueueResult();
        ReentrantLock transferCreateLock= getLock(transferId);
        try {
            transferCreateLock.lock();
            boolean exist = this.transferQueueMap.get(transferId) != null;
            if (exist) {
                createQueueResult.setTransferQueue(this.transferQueueMap.get(transferId));
                String[] elements = MetaInfo.INSTANCE_ID.split(":");
                createQueueResult.setPort(Integer.parseInt(elements[1]));
                createQueueResult.setRedirectIp(elements[0]);
                return createQueueResult;
            }
            if (MetaInfo.PROPERTY_DEPLOY_MODE.equals(DeployMode.cluster.name()) && !forceCreateLocal) {
                /*
                 * 缓存的集群信息中能够找到，直接返回信息
                 */
                if (this.transferQueueApplyInfoMap.get(transferId) != null) {
                    TransferQueueApplyInfo transferQueueApplyInfo = this.transferQueueApplyInfoMap.get(transferId);
                    if (!transferQueueApplyInfo.getInstanceId().equals(MetaInfo.INSTANCE_ID)) {
                        String instanceId = transferQueueApplyInfo.getInstanceId();
                        String[] args = instanceId.split(":");
                        String ip = args[0];
                        String portString = args[1];
                        createQueueResult.setPort(Integer.parseInt(portString));
                        createQueueResult.setRedirectIp(ip);
                        return createQueueResult;
                    } else {
                        /*
                         * 这种情况存在于本地已删除，而集群信息未同步更新，可能存在延迟，这时重走申请流程
                         */
                    }
                }

                Osx.Outbound applyTopicResponse = this.applyFromMaster(transferId, sessionId, MetaInfo.INSTANCE_ID);
                logger.info("apply topic response {}", applyTopicResponse);

                if (applyTopicResponse != null) {

                    /*
                     * 从clustermananger 返回的结果中比对instantceId ，如果为本实例，则在本地建Q
                     */
                    String applyInstanceId = applyTopicResponse.getMetadataMap().get(Osx.Metadata.InstanceId.name());

                    if (MetaInfo.INSTANCE_ID.equals(applyInstanceId)) {

                        String[] elements = MetaInfo.INSTANCE_ID.split(":");
                        createQueueResult.setPort(Integer.parseInt(elements[1]));
                        createQueueResult.setRedirectIp(elements[0]);
                        createQueueResult.setTransferQueue(localCreate(transferId, sessionId));
                        registerTransferQueue(transferId, sessionId);
                        //createQueueResult = applyFromCluster(transferId,sessionId);
                    } else {
                        if (applyInstanceId != null) {
                            String[] args = applyInstanceId.split(":");
                            String ip = args[0];
                            String portString = args[1];
                            int grpcPort = Integer.parseInt(portString);
                            createQueueResult.setRedirectIp(ip);
                            createQueueResult.setPort(grpcPort);
                        } else {
                            throw new CreateTopicErrorException("apply topic from master error");
                        }
                    }
                } else {
                    throw new RuntimeException();
                }
            } else {
                /*
                 * 单机版部署，直接本地建Q
                 */
                createQueueResult.setTransferQueue(localCreate(transferId, sessionId));
//                String[] args = MetaInfo.INSTANCE_ID.split("_");
//                String ip = args[0];
//                String portString = args[1];

                createQueueResult.setPort(MetaInfo.PROPERTY_GRPC_PORT);
                createQueueResult.setRedirectIp(NetUtils.getLocalHost());
            }
            return createQueueResult;
        } finally {
            transferCreateLock.unlock();

        }
    }


    private void registerTransferQueue(String transferId, String sessionId) {
        String path = buildZkPath(transferId);
        TransferQueueApplyInfo transferQueueApplyInfo = new TransferQueueApplyInfo();
        transferQueueApplyInfo.setTransferId(transferId);
        transferQueueApplyInfo.setSessionId(sessionId);
        transferQueueApplyInfo.setInstanceId(MetaInfo.INSTANCE_ID);
        transferQueueApplyInfo.setApplyTimestamp(System.currentTimeMillis());
        try {
            ServiceContainer.zkClient.create(path, JsonUtil.object2Json(transferQueueApplyInfo), true);
        } catch (KeeperException.NodeExistsException e) {
            logger.error("register path {} to zk error", path);
        }
    }

    public String buildZkPath(String transferId) {
        return ZK_QUEUE_PREFIX + "/" + transferId;
    }

    private CreateQueueResult applyFromCluster(String transferId, String sessionId) {
        CreateQueueResult createQueueResult = null;

        if (MetaInfo.PROPERTY_USE_ZOOKEEPER) {
            createQueueResult = new CreateQueueResult();
            String path = buildZkPath(transferId);
            boolean exist = ServiceContainer.zkClient.checkExists(path);
            if (exist) {
                String content = ServiceContainer.zkClient.getContent(path);
                TransferQueueApplyInfo transferQueueApplyInfo = JsonUtil.json2Object(content, TransferQueueApplyInfo.class);
            } else {
                /*
                 * 如何平均
                 */
                TransferQueueApplyInfo transferQueueApplyInfo = new TransferQueueApplyInfo();
                transferQueueApplyInfo.setTransferId(transferId);
                transferQueueApplyInfo.setSessionId(sessionId);
                transferQueueApplyInfo.setInstanceId(MetaInfo.INSTANCE_ID);
                transferQueueApplyInfo.setApplyTimestamp(System.currentTimeMillis());
                try {
                    ServiceContainer.zkClient.create(path, JsonUtil.object2Json(transferQueueApplyInfo), true);
                } catch (KeeperException.NodeExistsException e) {
                    logger.error("register path {} in zk error", path);
                }
                String content = ServiceContainer.zkClient.getContent(path);
                transferQueueApplyInfo = JsonUtil.json2Object(content, TransferQueueApplyInfo.class);
                assert transferQueueApplyInfo != null;
                if (MetaInfo.INSTANCE_ID.equals(transferQueueApplyInfo.getInstanceId())) {
                    createQueueResult.setTransferQueue(localCreate(transferId, sessionId));
                } else {
                    String[] elements = MetaInfo.INSTANCE_ID.split(":");
                    createQueueResult.setPort(Integer.parseInt(elements[1]));
                    createQueueResult.setRedirectIp(elements[0]);
                }
            }
        }
        return createQueueResult;
    }


    public Osx.Outbound applyFromMaster(String topic, String sessionId, String instanceId) {

        if (!isMaster()) {
            RouterInfo routerInfo = this.getMasterAddress();
            //context.setRouterInfo(routerInfo);
            ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo, true);
            PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub stub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
            try {
                Osx.Inbound.Builder builder = Osx.Inbound.newBuilder();
                builder.putMetadata(Osx.Metadata.MessageTopic.name(), topic);
                builder.putMetadata(Osx.Metadata.InstanceId.name(), instanceId);
                builder.putMetadata(Osx.Header.SessionID.name(), sessionId);
                builder.putMetadata(Osx.Header.TechProviderCode.name(), MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
                builder.putMetadata(Osx.Metadata.TargetMethod.name(), TargetMethod.APPLY_TOPIC.name());
                return stub.invoke(builder.build());
            } catch (StatusRuntimeException e) {
                logger.error("apply topic {} from master error", topic, e);
                throw new RemoteRpcException("send to " + routerInfo.toKey() + " error");
            }
        } else {
            TransferQueueApplyInfo transferQueueApplyInfo = this.handleClusterApply(topic,
                    instanceId, sessionId);
            Osx.Outbound.Builder outboundBuilder = Osx.Outbound.newBuilder();
            outboundBuilder.putMetadata(Osx.Metadata.MessageTopic.name(), topic);
            outboundBuilder.putMetadata(Osx.Metadata.InstanceId.name(), instanceId);
            outboundBuilder.putMetadata(Osx.Metadata.Timestamp.name(), Long.toString(transferQueueApplyInfo.getApplyTimestamp()));
            outboundBuilder.setCode(StatusCode.SUCCESS);
            outboundBuilder.setMessage(Dict.SUCCESS);
            return outboundBuilder.build();
        }
    }


    private RouterInfo getMasterAddress() {
        RouterInfo routerInfo = new RouterInfo();
        String[] args = MetaInfo.masterInfo.getInstanceId().split(Dict.COLON);
        routerInfo.setHost(args[0]);
        routerInfo.setPort(Integer.parseInt(args[1]));
        routerInfo.setProtocol(Protocol.grpc);
        return routerInfo;
    }


    private void unRegisterCluster(String transferId) {

        if (MetaInfo.isCluster() && MetaInfo.isCluster()) {
            logger.info("unRegister topic {} from zk", transferId);
            ServiceContainer.zkClient.delete(buildZkPath(transferId));
        }
    }

    private void setMsgCallBack(TransferQueue transferQueue) {
        this.msgCallBackRuleMap.forEach((rule, msgCallbacks) -> {

            if (rule.isMatch(transferQueue)) {
                //      logger.info("rule {} is mactched",rule);
                transferQueue.registerMsgCallback(msgCallbacks);
            } else {
                //        logger.info("rule {} is not matched",rule);
            }
        });
    }

    ;


    private TransferQueue localCreate(String topic, String sessionId) {
        logger.info("create local topic {}", topic);
        TransferQueue transferQueue = new TransferQueue(topic, this, MetaInfo.PROPERTY_TRANSFER_FILE_PATH_PRE + File.separator + MetaInfo.INSTANCE_ID);
        transferQueue.setSessionId(sessionId);
        transferQueue.start();
        transferQueue.registerDestoryCallback(() -> {
            this.transferQueueMap.remove(topic);
            if (this.sessionQueueMap.get(sessionId) != null) {
                this.sessionQueueMap.get(sessionId).remove(topic);
            }
            unRegisterCluster(topic);
        });
        setMsgCallBack(transferQueue);
        transferQueueMap.put(topic, transferQueue);
        sessionQueueMap.putIfAbsent(sessionId, new HashSet<>());
        sessionQueueMap.get(sessionId).add(topic);
        return transferQueue;
    }

    public TransferQueue getQueue(String topic) {
        return transferQueueMap.get(topic);
    }

    public Map<String, TransferQueue> getAllLocalQueue() {
        return this.transferQueueMap;
    }


    private void destroy(String topic) {
        Preconditions.checkArgument(StringUtils.isNotEmpty(topic));
        ReentrantLock transferIdLock = this.transferIdLockMap.get(topic);
        if (transferIdLock != null) {
            transferIdLock.lock();
        }
        try {
            TransferQueue transferQueue = getQueue(topic);
            if (transferQueue != null) {
                destroyInner(transferQueue);
                transferIdLockMap.remove(topic);
            }

        } finally {
            if (transferIdLock != null) {
                transferIdLock.unlock();
            }
        }
    }


    public void onError(String transferId, Throwable throwable) {
        TransferQueue transferQueue = transferQueueMap.get(transferId);
        if (transferQueue != null) {
            /*
             * 这里需要处理的问题是，当异常发生时，消费者并没有接入，等触发之后才接入
             */
            errorCallBackExecutor.execute(() -> transferQueue.onError(throwable));
        }
        this.destroy(transferId);
    }

    public void onCompleted(String transferId) {
        logger.info("transfer queue {} prepare to destory", transferId);
        TransferQueue transferQueue = transferQueueMap.get(transferId);
        if (transferQueue != null) {
            transferQueue.onCompeleted();
        }
        this.destroy(transferId);
        logger.info("transfer queue {} destoryed", transferId);
    }

    public TransferQueueApplyInfo queryGlobleQueue(String transferId) {
        return this.transferQueueApplyInfoMap.get(transferId);
    }

    public void destroyAll() {
        if (MetaInfo.isCluster()) {
            try {
                if (this.isMaster()) {
                    ServiceContainer.zkClient.delete(MASTER_PATH);
                }
                ServiceContainer.zkClient.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        this.transferQueueMap.forEach((transferId, transferQueue) -> {
            transferQueue.destory();
        });
    }


    public void addMsgCallBackRule(EventDriverRule rule, List<MsgEventCallback> callbacks) {
        this.msgCallBackRuleMap.put(rule, callbacks);
    }
}
