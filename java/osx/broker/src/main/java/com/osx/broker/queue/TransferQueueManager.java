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
import com.google.protobuf.ByteString;
import com.osx.broker.ServiceContainer;
import com.osx.core.config.MasterInfo;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.DeployMode;
import com.osx.core.constant.Dict;
import com.osx.core.constant.StatusCode;
import com.osx.core.constant.TransferStatus;
import com.osx.core.context.Context;
import com.osx.core.exceptions.CreateTopicErrorException;
import com.osx.core.exceptions.RemoteRpcException;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.frame.ServiceThread;
import com.osx.core.ptp.TargetMethod;
import com.osx.core.router.RouterInfo;
import com.osx.core.utils.JsonUtil;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import org.apache.commons.lang3.StringUtils;
import org.apache.zookeeper.KeeperException;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

public class TransferQueueManager {
    final String ZK_QUEUE_PREFIX = "/FATE-TRANSFER/QUEUE";
    final String MASTER_PATH = "/FATE-TRANSFER/MASTER";
    final String ZK_COMPONENTS_PREFIX = "/FATE-COMPONENTS/firework";
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
    ConcurrentHashMap<String, ReentrantLock> transferIdLockMap = new ConcurrentHashMap();
    volatile long transferApplyInfoVersion = -1;
    private ServiceThread cleanTask = new ServiceThread() {
        @Override
        public void run() {
            while (true) {
                this.waitForRunning(1000);
                checkAndClean();
            }
        }

        @Override
        public String getServiceName() {
            return "TransferQueueCleanTask";
        }
    };


    public TransferQueueManager() {
        instanceIds.add(MetaInfo.INSTANCE_ID);
        if (MetaInfo.isCluster()) {
            boolean pathExists = ServiceContainer.zkClient.checkExists(ZK_QUEUE_PREFIX);
            if (!pathExists) {
                ServiceContainer.zkClient.create(ZK_QUEUE_PREFIX, false);
            }
            List<String> initApplyInfo = ServiceContainer.zkClient.addChildListener(ZK_QUEUE_PREFIX, (path, children) -> {
                parseApplyInfo(children);
            });
            parseApplyInfo(initApplyInfo);
            ServiceContainer.zkClient.create(ZK_COMPONENTS_PREFIX + "/" + MetaInfo.INSTANCE_ID, true);
            List<String> initInstanceIds = ServiceContainer.zkClient.addChildListener(ZK_COMPONENTS_PREFIX, (path, children) -> {
                handleClusterInstanceId(children);
            });
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
     * @param instanceId
     * @return
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
                Integer count = temp.get(instanceId);
                if (count == null) {
                    temp.put(instanceId, 1);
                } else {
                    temp.put(instanceId, count + 1);
                }
                ;
            }
        });
        this.clusterTransferQueueCountMap = temp;
        masterQueueApplyInfoMap.forEach((k, v) -> {
            if (current - v.getApplyTimestamp() > 10000) {
                if (transferQueueApplyInfoMap.get(k) == null) {
                    masterQueueApplyInfoMap.remove(k);
                }
            };
        });
    }

    private MasterInfo parseMasterInfo(String masterContent) {
        MasterInfo masterInfo = JsonUtil.json2Object(masterContent, MasterInfo.class);
        return masterInfo;
    }

    private void handleClusterInstanceId(List<String> children) {
        this.instanceIds.clear();
        this.instanceIds.addAll(children);
        if(logger.isInfoEnabled()) {
            logger.info("instance change : {}", instanceIds);
        }
    }

    private synchronized void parseApplyInfo(List<String> children) {
        Set childSet = Sets.newHashSet(children);
        Set<String> intersecitonSet = Sets.intersection(transferQueueApplyInfoMap.keySet(), childSet);
        Set<String> needAddSet = null;
        if (intersecitonSet != null)
            needAddSet = Sets.difference(childSet, intersecitonSet);
        Set<String> needRemoveSet = Sets.difference(transferQueueApplyInfoMap.keySet(), intersecitonSet);
        if(logger.isInfoEnabled()) {
            logger.info("cluster apply info add {} remove {}", needAddSet, needRemoveSet);
        }
        if (needRemoveSet != null) {
            needRemoveSet.forEach(k -> {
                transferQueueApplyInfoMap.remove(k);
            });
        }
        if (needAddSet != null) {
            needAddSet.forEach(k -> {
                try {
                    String content = ServiceContainer.zkClient.getContent(ZK_QUEUE_PREFIX + "/" + k);
                    TransferQueueApplyInfo transferQueueApplyInfo = JsonUtil.json2Object(content, TransferQueueApplyInfo.class);
                    if (transferQueueApplyInfo != null) {
                        transferQueueApplyInfoMap.put(k, transferQueueApplyInfo);
                    }
                } catch (Exception e) {
                   logger.error("parse apply info from zk error",e);
                }
            });
        }
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
        transferQueueMap.forEach((transferId, transferQueue) -> {
            try {
                long lastReadTimestamp = transferQueue.getLastReadTimestamp();
                long lastWriteTimestamp = transferQueue.getLastWriteTimestamp();
                long freeTime = now - (lastReadTimestamp > lastWriteTimestamp ? lastReadTimestamp : lastWriteTimestamp);
                if (transferQueue.getTransferStatus() == TransferStatus.ERROR || transferQueue.getTransferStatus() == TransferStatus.FINISH) {
                    destroy(transferId);
                }
                if (freeTime > MetaInfo.PRPPERTY_QUEUE_MAX_FREE_TIME) {
                    if(logger.isInfoEnabled()) {
                        logger.info("transfer queue : {} freetime  {} need to be destroy", transferId, freeTime);
                    }
                    destroy(transferId);
                    return;
                }
            } catch (Exception igrone) {

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
            String intanceId = doClusterBalance(transferId, instanceId, sessionId);
            newTransferQueueApplyInfo.setTransferId(transferId);
            newTransferQueueApplyInfo.setInstanceId(instanceId);
            newTransferQueueApplyInfo.setSessionId(sessionId);
            newTransferQueueApplyInfo.setApplyTimestamp(current);
            this.masterQueueApplyInfoMap.put(transferId, newTransferQueueApplyInfo);
            return newTransferQueueApplyInfo;
        }
    }


    public CreateQueueResult createNewQueue(String transferId, String sessionId, boolean forceCreateLocal) {
        Preconditions.checkArgument(StringUtils.isNotEmpty(transferId));
        CreateQueueResult createQueueResult = new CreateQueueResult();
        ReentrantLock transferCreateLock = transferIdLockMap.get(transferId);
        if (transferCreateLock == null) {
            transferIdLockMap.putIfAbsent(transferId, new ReentrantLock(false));
        }
        transferCreateLock = transferIdLockMap.get(transferId);
        transferCreateLock.lock();
        try {

            boolean exist = this.transferQueueMap.get(transferId) != null ? true : false;
            if (exist) {
                createQueueResult.setTransferQueue(this.transferQueueMap.get(transferId));
                String[] elements = MetaInfo.INSTANCE_ID.split(":");
                createQueueResult.setPort(Integer.parseInt(elements[1]));
                createQueueResult.setRedirectIp(elements[0]);
                return createQueueResult;
            }
            if (MetaInfo.PROPERTY_DEPLOY_MODE.equals(DeployMode.cluster.name()) && !forceCreateLocal) {
                /**
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
                        /**
                         * 这种情况存在于本地已删除，而集群信息未同步更新，可能存在延迟，这时重走申请流程
                         */
                    }
                };

                Osx.Outbound applyTopicResponse = this.applyFromMaster(transferId,sessionId,MetaInfo.INSTANCE_ID);
                logger.info("apply topic response {}", applyTopicResponse);

                if (applyTopicResponse != null) {

                    /**
                     * 从clustermananger 返回的结果中比对instantceId ，如果为本实例，则在本地建Q
                     */
                    String  applyInstanceId = applyTopicResponse.getMetadataMap().get(Osx.Metadata.InstanceId.name());

                    if (MetaInfo.INSTANCE_ID.equals(applyInstanceId)) {

                        String[] elements = MetaInfo.INSTANCE_ID.split(":");
                        createQueueResult.setPort(Integer.parseInt(elements[1]));
                        createQueueResult.setRedirectIp(elements[0]);
                        createQueueResult.setTransferQueue(localCreate(transferId, sessionId));
                        registerTransferQueue(transferId, sessionId);
                        //createQueueResult = applyFromCluster(transferId,sessionId);
                    } else {
                        if(applyInstanceId!=null) {
                            String[] args = applyInstanceId.split(":");
                            String ip = args[0];
                            String portString = args[1];
                            int grpcPort = Integer.parseInt(portString);
                            createQueueResult.setRedirectIp(ip);
                            createQueueResult.setPort(grpcPort);
                        }else{
                            throw new CreateTopicErrorException("apply topic from master error");
                        }
                    };
                } else {
                    throw new RuntimeException();
                }
            } else {
                /**
                 * 单机版部署，直接本地建Q
                 */
                createQueueResult.setTransferQueue(localCreate(transferId, sessionId));
                String[] args = MetaInfo.INSTANCE_ID.split(":");
                String ip = args[0];
                String portString = args[1];
                createQueueResult.setPort(Integer.parseInt(portString));
                createQueueResult.setRedirectIp(ip);
            }
            return createQueueResult;
        } finally {
            transferCreateLock.unlock();
        }
    }


    private void registerTransferQueue(String transferId, String sessionId) {
        StringBuffer sb = new StringBuffer();
        sb.append(ZK_QUEUE_PREFIX).append("/");
        sb.append(transferId);
        String path = sb.toString();
        TransferQueueApplyInfo transferQueueApplyInfo = new TransferQueueApplyInfo();
        transferQueueApplyInfo.setTransferId(transferId);
        transferQueueApplyInfo.setSessionId(sessionId);
        transferQueueApplyInfo.setInstanceId(MetaInfo.INSTANCE_ID);
        transferQueueApplyInfo.setApplyTimestamp(System.currentTimeMillis());
        try {
            ServiceContainer.zkClient.create(path, JsonUtil.object2Json(transferQueueApplyInfo), true);
        } catch (KeeperException.NodeExistsException e) {
            e.printStackTrace();
        }
    }

    private CreateQueueResult applyFromCluster(String transferId, String sessionId) {
        CreateQueueResult createQueueResult = null;

        if (MetaInfo.PROPERTY_USE_ZOOKEEPER) {
            createQueueResult = new CreateQueueResult();
            StringBuffer sb = new StringBuffer();
            sb.append(ZK_QUEUE_PREFIX).append("/");
            sb.append(transferId);
            String path = sb.toString();
            boolean exist = ServiceContainer.zkClient.checkExists(path);
            if (exist) {
                String content = ServiceContainer.zkClient.getContent(path);
                TransferQueueApplyInfo transferQueueApplyInfo = JsonUtil.json2Object(content, TransferQueueApplyInfo.class);
            } else {
                /**
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
                    e.printStackTrace();
                }
                String content = ServiceContainer.zkClient.getContent(path);
                transferQueueApplyInfo = JsonUtil.json2Object(content, TransferQueueApplyInfo.class);
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


    public Osx.Outbound applyFromMaster( String topic,String sessionId,String instanceId) {
        if (!isMaster()) {

            RouterInfo  routerInfo = this.getMasterAddress();
            //context.setRouterInfo(routerInfo);
            ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo,true);
            PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub stub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
            try {
                Osx.Inbound.Builder   builder = Osx.Inbound.newBuilder();
                builder.putMetadata(Osx.Metadata.MessageTopic.name(),topic);
                builder.putMetadata(Osx.Metadata.InstanceId.name(),instanceId);
                builder.putMetadata(Osx.Header.SessionID.name(),sessionId);
                return stub.invoke(builder.build());
            }catch(StatusRuntimeException e){
                throw  new RemoteRpcException("send to "+routerInfo.toKey()+" error");
            }
        } else {
            TransferQueueApplyInfo transferQueueApplyInfo = this.handleClusterApply(topic,
                    instanceId, sessionId);
            Osx.Outbound.Builder outboundBuilder = Osx.Outbound.newBuilder();
            outboundBuilder.getMetadataMap().put(Osx.Metadata.MessageTopic.name(), topic);
            outboundBuilder.getMetadataMap().put(Osx.Metadata.InstanceId.name(), instanceId);
            outboundBuilder.getMetadataMap().put(Osx.Metadata.Timestamp.name(), Long.toString(transferQueueApplyInfo.getApplyTimestamp()));
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
        return routerInfo;
    }


    private void unRegisterCluster(String transferId) {
        logger.info("unRegister transferId {}", transferId);
        if (MetaInfo.isCluster()) {
            ServiceContainer.zkClient.delete(ZK_QUEUE_PREFIX + "/" + transferId);
        }
    }


    private TransferQueue localCreate(String transferId, String sessionId) {
        logger.info("create local topic {}",transferId);
        TransferQueue transferQueue = new TransferQueue(transferId, this, MetaInfo.PROPERTY_TRANSFER_FILE_PATH_PRE + File.separator + MetaInfo.INSTANCE_ID);
        transferQueue.setSessionId(sessionId);
        transferQueue.start();
        transferQueue.registeDestoryCallback(() -> {
            this.transferQueueMap.remove(transferId);
            if (this.sessionQueueMap.get(sessionId) != null) {
                this.sessionQueueMap.get(sessionId).remove(transferId);
            }
        });
        transferQueueMap.put(transferId, transferQueue);
        sessionQueueMap.putIfAbsent(sessionId, new HashSet<>());
        sessionQueueMap.get(sessionId).add(transferId);
        return transferQueue;
    }

    public TransferQueue getQueue(String transferId) {
        return transferQueueMap.get(transferId);
    }

    public Map<String, TransferQueue> getAllLocalQueue() {
        return this.transferQueueMap;
    }


    private void destroy(String transferId) {
        Preconditions.checkArgument(StringUtils.isNotEmpty(transferId));
        ReentrantLock transferIdLock = this.transferIdLockMap.get(transferId);
        if (transferIdLock != null) {
            transferIdLock.lock();
        }
        try {
            TransferQueue transferQueue = getQueue(transferId);
            if (transferQueue != null) {
                destroyInner(transferQueue);
                transferIdLockMap.remove(transferId);
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
            /**
             * 这里需要处理的问题是，当异常发生时，消费者并没有接入，等触发之后才接入
             */
            errorCallBackExecutor.execute(() -> {
                transferQueue.onError(throwable);
            });
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
        logger.info("prepare to destory {}", transferQueueMap);
        if (MetaInfo.isCluster()) {
            try {
                if (this.isMaster()) {
                    ServiceContainer.zkClient.delete(MASTER_PATH);
                }
                ServiceContainer.zkClient.close();
                ;
            } catch (Exception e) {
                e.printStackTrace();
            }
            logger.info("unregister component over");
        }
        this.transferQueueMap.forEach((transferId, transferQueue) -> {
            transferQueue.destory();
        });
    }
}
