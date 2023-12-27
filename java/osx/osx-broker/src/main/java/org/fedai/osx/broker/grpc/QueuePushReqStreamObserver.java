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
package org.fedai.osx.broker.grpc;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.command.Command;
import com.webank.eggroll.core.meta.Meta;
import com.webank.eggroll.core.transfer.Transfer;
import com.webank.eggroll.core.transfer.TransferServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.broker.eggroll.*;
import org.fedai.osx.broker.queue.TransferQueueManager;
import org.fedai.osx.broker.router.RouterService;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.ActionType;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.constant.TransferStatus;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.context.Protocol;
import org.fedai.osx.core.exceptions.*;
import org.fedai.osx.core.frame.GrpcConnectionFactory;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.utils.FlowLogUtil;
import org.fedai.osx.core.utils.ToStringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class QueuePushReqStreamObserver implements StreamObserver<Proxy.Packet> {

    static public ConcurrentHashMap<Integer, QueuePushReqStreamObserver> queueIdMap = new ConcurrentHashMap<>();
    static AtomicInteger seq = new AtomicInteger(0);
    Logger logger = LoggerFactory.getLogger(QueuePushReqStreamObserver.class);
    OsxContext context;
    ErRollSiteHeader rsHeader = null;
    TransferStatus transferStatus = TransferStatus.INIT;
    CountDownLatch finishLatch = new CountDownLatch(1);
    StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> putBatchSinkPushReqSO;
    RouterInfo routerInfo;
    Proxy.Metadata metadata;
    String brokerTag;
    private boolean isDst = false;

    private StreamObserver<Proxy.Packet> forwardPushReqSO;
    private StreamObserver<Proxy.Metadata> backRespSO;
    private RouterService routerService;
    private TransferQueueManager transferQueueManager;
    private ManagedChannel channel;

    public QueuePushReqStreamObserver(OsxContext context, RouterService routerService, TransferQueueManager transferQueueManager,
                                      StreamObserver backRespSO
    ) {
        this.context = context;
        this.routerService = routerService;
        this.backRespSO = backRespSO;
        this.context.setServiceName("pushTransfer");
        this.transferQueueManager = transferQueueManager;
    }

    public StreamObserver<Proxy.Packet> getForwardPushReqSO() {
        return forwardPushReqSO;
    }

    public void setForwardPushReqSO(StreamObserver<Proxy.Packet> forwardPushReqSO) {
        this.forwardPushReqSO = forwardPushReqSO;
    }

    public void init(Proxy.Packet packet) throws Exception {
        TransferUtil.assableContextFromProxyPacket(context, packet);
        Proxy.Metadata metadata = packet.getHeader();
        String desPartyId = context.getDesNodeId();
        String srcPartyId = context.getSrcNodeId();
        ByteString encodedRollSiteHeader = metadata.getExt();
        rsHeader = ErRollSiteHeader.parseFromPb(Transfer.RollSiteHeader.parseFrom(encodedRollSiteHeader));
        Integer partitionId = rsHeader.getPartitionId();
        brokerTag = "putBatch-" + rsHeader.getRsKey("#", "__rsk") + "-" + partitionId;
        context.setSessionId(rsHeader.getRollSiteSessionId());
        context.setTopic(brokerTag);

        if (MetaInfo.PROPERTY_SELF_PARTY.contains(desPartyId)) {
            isDst = true;
        }
        /**
         * 检查目的地是否为自己
         */
        if (!isDst) {
            routerInfo = routerService.route(context.getSrcNodeId(), context.getSrcComponent(), context.getDesNodeId(), context.getDesComponent());
            if (routerInfo == null) {
                logger.error("no router info is found for party id {}", context.getDesNodeId());
                throw new NoRouterInfoException("no router is found for party id" + context.getDesNodeId());
            }
        }
        if (isDst) {
            if (MetaInfo.PROPERTY_OPEN_MOCK_EGGPAIR) {
                mockEggroll(context, packet);
            } else {
                initEggroll(context, packet);
            }
        } else {
            context.setActionType(ActionType.PUSH_REMOTE.name());
            context.setRouterInfo(routerInfo);
            context.setSrcNodeId(routerInfo.getSourcePartyId());
            context.setDesNodeId(routerInfo.getDesPartyId());
            if (routerInfo.getProtocol().equals(Protocol.http)) {
                logger.error("invalid router info {}, grpc stream is not support http1.x", routerInfo);
                throw new SysException("invalid router info for grpc stream");
            } else {
                ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(context.getRouterInfo());
                DataTransferServiceGrpc.DataTransferServiceStub stub = DataTransferServiceGrpc.newStub(managedChannel);
                ForwardPushRespSO forwardPushRespSO = new ForwardPushRespSO(context, backRespSO, () -> {
                    finishLatch.countDown();
                }, (t) -> {
                    finishLatch.countDown();
                });
                forwardPushReqSO = stub.push(forwardPushRespSO);
            }
        }
        transferStatus = TransferStatus.TRANSFERING;
    }

    private void mockEggroll(OsxContext context, Proxy.Packet firstRequest) {
        metadata = firstRequest.getHeader();
        routerInfo = new RouterInfo();
        context.setRouterInfo(routerInfo);
        routerInfo.setHost(MetaInfo.PROPERTY_MOCK_EGGPAIR_IP);
        routerInfo.setPort(MetaInfo.PROPERTY_MOCK_EGGPAIR_PORT);
        context.setSrcNodeId(routerInfo.getSourcePartyId());
        context.setDesNodeId(MetaInfo.PROPERTY_MOCK_EGGPAIR_PARTYID);
        ManagedChannel channel = GrpcConnectionFactory.createManagedChannel(routerInfo);
        TransferServiceGrpc.TransferServiceStub stub = TransferServiceGrpc.newStub(channel);
        CompletableFuture<ErTask> commandFuture = new CompletableFuture<>();
        commandFuture.complete(new ErTask());
        putBatchSinkPushReqSO = stub.send(new PutBatchSinkPushRespSO(metadata, commandFuture, backRespSO, finishLatch, routerInfo));
    }

    private void initEggroll(OsxContext context, Proxy.Packet firstRequest) {
        if (StringUtils.isEmpty(MetaInfo.PROPERTY_EGGROLL_CLUSTER_MANANGER_IP)) {
            throw new SysException("eggroll cluter manager ip is not found");
        }

        metadata = firstRequest.getHeader();
        String oneLineStringMetadata = ToStringUtils.toOneLineString(metadata);
        context.setActionType(ActionType.PUSH_EGGPAIR.name());
        String rsKey = rsHeader.getRsKey("#", "__rsk");
        String sessionId = String.join("_", rsHeader.getRollSiteSessionId(), rsHeader.getDstRole(), rsHeader.getDstPartyId());
        context.setSessionId(sessionId);
        ErSession session = null;
        try {
            session = PutBatchSinkUtil.sessionCache.get(sessionId);
        } catch (ExecutionException e) {
            logger.error("get session error ", e);
        }
        if (!SessionStatus.ACTIVE.name().equals(session.getErSessionMeta().getStatus())) {
            SessionInitException error = new SessionInitException("eggroll session " + sessionId + " invalid status : " + session.getErSessionMeta().getStatus());
            onError(error);
            throw error;
        }

        String namespace = rsHeader.getRollSiteSessionId();
        String name = rsKey;
        RollPairContext ctx = new RollPairContext(session);
        Map rpOptions = Maps.newHashMap();
        rpOptions.putAll(rsHeader.getOptions());
        rpOptions.put(Dict.TOTAL_PARTITIONS_SNAKECASE, rsHeader.getTotalPartitions().toString());

        if (rsHeader.getDataType().equals("object")) {
            rpOptions.put(Dict.SERDES, SerdesTypes.EMPTY.name());
        } else {
            rpOptions.put(Dict.SERDES, rsHeader.getOptions().getOrDefault("serdes", SerdesTypes.PICKLE.name()));
        }

        // use in-memory store here
        rpOptions.put(Dict.STORE_TYPE_SNAKECASE, "IN_MEMORY");

        // table creates here
        RollPair rp = ctx.load(namespace, name, rpOptions);
        Integer partitionId = rsHeader.getPartitionId();
        ErPartition partition = rp.getStore().getPartition(partitionId);
        ErProcessor egg = ctx.getErSession().routeToEgg(partition);
        String jobId = IdUtils.generateJobId(ctx.getErSession().getSessionId(), brokerTag, "-");
        Map<String, String> jobOptions = new HashMap<>();

        jobOptions.putAll(rsHeader.getOptions());
        jobOptions.put(SessionConfKeys.CONFKEY_SESSION_ID, ctx.getErSession().getSessionId());
        ErJob job = new ErJob(
                jobId,
                RollPair.PUT_BATCH,
                Lists.newArrayList(new ErJobIO(rp.getStore(), new ErSerdes(0), new ErSerdes(0), new ErPartitioner(0))),
                Lists.newArrayList(new ErJobIO(rp.getStore(), new ErSerdes(0), new ErSerdes(0), new ErPartitioner(0))),
                Lists.newArrayList(),
                jobOptions);

        ErTask task = new ErTask(brokerTag,
                RollPair.PUT_BATCH,
                Lists.newArrayList(partition),
                Lists.newArrayList(partition),
                job);

        Future<ErTask> commandFuture = RollPairContext.executor.submit(() -> {
            CommandClient commandClient = new CommandClient(egg.getCommandEndpoint());
            Command.CommandResponse commandResponse = commandClient.call(RollPair.EGG_RUN_TASK_COMMAND, task);
            Meta.Task taskMeta = Meta.Task.parseFrom(commandResponse.getResultsList().get(0));
            return ErTask.parseFromPb(taskMeta);
        });
        RouterInfo routerInfo = new RouterInfo();
        routerInfo.setProtocol(Protocol.grpc);
        context.setRouterInfo(routerInfo);
        routerInfo.setHost(egg.getTransferEndpoint().getHost());
        routerInfo.setPort(egg.getTransferEndpoint().getPort());
        context.setSrcNodeId(routerInfo.getSourcePartyId());
        context.setDesNodeId(routerInfo.getDesPartyId());
        ManagedChannel channel = GrpcConnectionFactory.createManagedChannel(routerInfo);
        TransferServiceGrpc.TransferServiceStub stub = TransferServiceGrpc.newStub(channel);
        putBatchSinkPushReqSO = stub.send(new PutBatchSinkPushRespSO(metadata, commandFuture, backRespSO, finishLatch, routerInfo));
    }


    @Override
    public void onNext(Proxy.Packet value) {
        try {
            if (value.getHeader() != null) {
                context.setTraceId(Long.toString(value.getHeader().getSeq()));
            }
//            long seq = value.getHeader().getSeq();
            context.setDataSize(value.getSerializedSize());
//            context.setCaseId(Long.toString(seq));
            if (transferStatus.equals(TransferStatus.INIT)) {
                init(value);
            }
            if (transferStatus.equals(TransferStatus.TRANSFERING)) {
                if (isDst) {
                    context.setActionType(ActionType.PUSH_EGGPAIR.name());
                    Transfer.TransferHeader.Builder transferHeaderBuilder = Transfer.TransferHeader.newBuilder();
                    Transfer.TransferHeader tbHeader = transferHeaderBuilder.setId((int) metadata.getSeq())
                            .setTag(brokerTag)
                            .setExt(value.getHeader().getExt()).build();
                    Transfer.TransferBatch.Builder transferBatchBuilder = Transfer.TransferBatch.newBuilder();
                    Transfer.TransferBatch tbBatch = transferBatchBuilder.setHeader(tbHeader)
                            .setData(value.getBody().getValue())
                            .build();
                    putBatchSinkPushReqSO.onNext(tbBatch);
                } else {
                    forwardPushReqSO.onNext(value);
                }
            }
        } catch (Exception e) {
            logger.error("push error1", e);
            ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
            context.setException(e);
            context.setReturnCode(exceptionInfo.getCode());
            throw ErrorMessageUtil.toGrpcRuntimeException(e);

        } finally {
            FlowLogUtil.printFlowLog(context);
        }
    }


    @Override
    public void onError(Throwable t) {
        /**
         * 传递错误
         */
        logger.info("onError", t);
        context.setException(t);
        // TODO: 2021/12/21这里需要补充逻辑
        /**
         * 1.停止消费者，需要考虑产生异常时消费者并没有接入。
         * 2.销毁队列
         */
        if (isDst) {
            putBatchSinkPushReqSO.onError(t);
        } else {
            if (forwardPushReqSO != null) {
                forwardPushReqSO.onError(t);
            }
        }
    }

    @Override
    public void onCompleted() {
        if (isDst) {
            if (putBatchSinkPushReqSO != null) {
                putBatchSinkPushReqSO.onCompleted();
            }
        } else {
            if (forwardPushReqSO != null) {
                forwardPushReqSO.onCompleted();
                try {
                    if (!finishLatch.await(MetaInfo.PROPERTY_GRPC_ONCOMPLETED_WAIT_TIMEOUT, TimeUnit.SECONDS)) {
                        onError(new TimeoutException());
                    }
                } catch (InterruptedException e) {
                    onError(e);
                }
            }
        }
    }

}
