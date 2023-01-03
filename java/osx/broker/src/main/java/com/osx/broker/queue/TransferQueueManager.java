package com.osx.broker.queue;

import com.firework.cluster.rpc.Firework;
import com.firework.cluster.rpc.FireworkServiceGrpc;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.osx.broker.ServiceContainer;
import com.osx.core.config.MasterInfo;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.DeployMode;
import com.osx.core.constant.Dict;
import com.osx.core.constant.StatusCode;
import com.osx.core.constant.TransferStatus;
import com.osx.core.context.Context;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.frame.ServiceThread;
import com.osx.core.router.RouterInfo;
import com.osx.core.utils.JsonUtil;
import io.grpc.ManagedChannel;
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
                    logger.info("===============master  remove {}", k);
                    masterQueueApplyInfoMap.remove(k);
                }
            }
            ;
        });
        // logger.info("==========cluster count info ============{}",clusterTransferQueueCountMap);
    }

    private MasterInfo parseMasterInfo(String masterContent) {
        MasterInfo masterInfo = JsonUtil.json2Object(masterContent, MasterInfo.class);
        return masterInfo;
    }

    private void handleClusterInstanceId(List<String> children) {
        this.instanceIds.clear();
        this.instanceIds.addAll(children);
        logger.info("instance change : {}", instanceIds);
    }

    private synchronized void parseApplyInfo(List<String> children) {
        Set childSet = Sets.newHashSet(children);
        Set<String> intersecitonSet = Sets.intersection(transferQueueApplyInfoMap.keySet(), childSet);
        Set<String> needAddSet = null;
        if (intersecitonSet != null)
            needAddSet = Sets.difference(childSet, intersecitonSet);
        Set<String> needRemoveSet = Sets.difference(transferQueueApplyInfoMap.keySet(), intersecitonSet);
        logger.info("cluster apply info add {} remove {}", needAddSet, needRemoveSet);
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
                    e.printStackTrace();
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
                        e.printStackTrace();
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
                e.printStackTrace();
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
                //    logger.info("transfer queue: {} status: {}  write finish: {} cost time {}",transferId,transferQueue.getTransferStatus(),transferQueue.isWriteOver(),costTime);
                if (transferQueue.getTransferStatus() == TransferStatus.ERROR || transferQueue.getTransferStatus() == TransferStatus.FINISH) {
                    destroy(transferId);
                }
                if (freeTime > MetaInfo.PRPPERTY_QUEUE_MAX_FREE_TIME) {
                    logger.info("transfer queue : {} freetime  {} need to be destroy", transferId, freeTime);
                    destroy(transferId);
                    return;
                }
            } catch (Exception igrone) {
                igrone.printStackTrace();
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
                }
                ;


                Firework.ApplyTransferQueueRequest request = Firework.ApplyTransferQueueRequest.newBuilder().
                        setTransferId(transferId).
                        setInstanceId(MetaInfo.INSTANCE_ID).
                        setSessionId(sessionId).build();


                Firework.ApplyTransferQueueResponse applyTransferQueueResponse = this.applyFromMaster(request);
                logger.info("apply transfer queue response {}", applyTransferQueueResponse);

                if (applyTransferQueueResponse != null) {
                    int applyReturnCode = applyTransferQueueResponse.getCode();

                    //  TransferQueueApplyInfo transferQueueApplyInfo=applyTransferQueueResponse.getTransferQueueApplyInfo();

                    /**
                     * 从clustermananger 返回的结果中比对instantceId ，如果为本实例，则在本地建Q
                     */
                    if (MetaInfo.INSTANCE_ID.equals(applyTransferQueueResponse.getInstanceId())) {

                        String[] elements = MetaInfo.INSTANCE_ID.split(":");
                        createQueueResult.setPort(Integer.parseInt(elements[1]));
                        createQueueResult.setRedirectIp(elements[0]);
                        createQueueResult.setTransferQueue(localCreate(transferId, sessionId));
                        registerTransferQueue(transferId, sessionId);
                        //createQueueResult = applyFromCluster(transferId,sessionId);

                    } else {
                        String instanceId = applyTransferQueueResponse.getInstanceId();
                        String[] args = instanceId.split(":");
                        String ip = args[0];
                        String portString = args[1];
                        int grpcPort = Integer.parseInt(portString);
                        createQueueResult.setRedirectIp(ip);
                        createQueueResult.setPort(grpcPort);
                    }
                    ;
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


    public Osx.Outbound applyFromMaster(Context context, Osx.Inbound inbound) {
        if (!isMaster()) {
            ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(this.getMasterAddress(),true);
            PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub stub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
            return stub.invoke(inbound);
        } else {

            String topic = inbound.getMetadataMap().get(Osx.Metadata.MessageTopic);
            String instanceId = inbound.getMetadataMap().get(Osx.Metadata.InstanceId);
            String sessionId = inbound.getMetadataMap().get(Osx.Header.SessionID);
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


    public Firework.ApplyTransferQueueResponse applyFromMaster(Firework.ApplyTransferQueueRequest produceRequest) {
        if (!isMaster()) {
            ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(getMasterAddress(),true);
            FireworkServiceGrpc.FireworkServiceBlockingStub stub = FireworkServiceGrpc.newBlockingStub(managedChannel);
            return stub.applyTransferQueue(produceRequest);
        } else {
            TransferQueueApplyInfo transferQueueApplyInfo = this.handleClusterApply(produceRequest.getTransferId(),
                    produceRequest.getInstanceId(), produceRequest.getSessionId());
            Firework.ApplyTransferQueueResponse.Builder resultBuilder = Firework.ApplyTransferQueueResponse.newBuilder();
            resultBuilder.setTransferId(transferQueueApplyInfo.getTransferId());
            resultBuilder.setApplyTimestamp(transferQueueApplyInfo.getApplyTimestamp());
            resultBuilder.setCode(0);
            resultBuilder.setInstanceId(transferQueueApplyInfo.getInstanceId());
            return resultBuilder.build();
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
//        ManagedChannel managedChannel = GrpcConnectionPool.getPool().getManagedChannel(MetaInfo.getClusterManagerHost(), MetaInfo.getClusterManagerPort());
//        FireworkServiceGrpc.FireworkServiceBlockingStub stub = FireworkServiceGrpc.newBlockingStub(managedChannel);
//        Firework.UnRegisterTransferQueueRequest.Builder  unRegisterTransferQueueRequestBuilder = Firework.UnRegisterTransferQueueRequest.newBuilder();
//        unRegisterTransferQueueRequestBuilder.setTransferId(transferId);
//        stub.unRegisterTransferQueue(unRegisterTransferQueueRequestBuilder.build());
        if (MetaInfo.isCluster()) {
            ServiceContainer.zkClient.delete(ZK_QUEUE_PREFIX + "/" + transferId);
        }


    }


    private TransferQueue localCreate(String transferId, String sessionId) {
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
                    System.err.println("=========quit master");
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
            System.err.println("kkkkkkkkkkkkkkkkk");
        });

        logger.info("over========");
    }

//    public synchronized FireworkTransfer.SyncTransferInfoResponse syncTransferQueueApplyInfo(FireworkTransfer.SyncTransferInfoRequest syncTransferInfoRequest){
//       //logger.info("============================= setTransferQueueApplyInfo {}",data);
//        FireworkTransfer.SyncTransferInfoResponse.Builder  syncTransferInfoResponseBuilder = FireworkTransfer.SyncTransferInfoResponse.newBuilder();
//        syncTransferInfoResponseBuilder.setPreVersion(transferApplyInfoVersion);
//        if(syncTransferInfoRequest!=null) {
//            if (syncTransferInfoRequest.getVersion()>transferApplyInfoVersion) {
//                syncTransferInfoResponseBuilder.setCode(StatusCode.SUCCESS);
//                if(syncTransferInfoRequest.getData()!=null) {
//                    byte[] dataBytes = syncTransferInfoRequest.getData().toByteArray();
//                    Map tempData = JsonUtil.json2Object(dataBytes, Map.class);
//                    if(tempData!=null){
//                        Map<String, TransferQueueApplyInfo>  resultData = Maps.newHashMap();
//                        tempData.forEach((transferId,mapData)->{
//                            resultData.put(transferId.toString(), JsonUtil.json2Object(JsonUtil.object2Json(mapData),TransferQueueApplyInfo.class));
//                        });
//                        this.transferQueueApplyInfoMap = resultData;
//                    }
//                    transferApplyInfoVersion = syncTransferInfoRequest.getVersion();
//                    List ids  =  syncTransferInfoRequest.getInstanceIdsList();
//                    instanceIds.clear();
//                    instanceIds.addAll(ids);
//                    instanceIds.add(MetaInfo.INSTANCE_ID);
//                }
//            }else{
//                syncTransferInfoResponseBuilder.setCode(StatusCode.TRANSFER_APPLYINFO_SYNC_ERROR);
//                syncTransferInfoResponseBuilder.setMsg("version "+syncTransferInfoRequest.getVersion()+" is expired ,now is "+transferApplyInfoVersion);
//            }
//        }
//        return  syncTransferInfoResponseBuilder.build();
//    }


}
