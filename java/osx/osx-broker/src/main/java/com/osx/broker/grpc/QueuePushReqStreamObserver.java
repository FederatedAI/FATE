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
package com.osx.broker.grpc;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.osx.api.constants.Protocol;
import com.osx.api.context.Context;
import com.osx.api.router.RouterInfo;
import com.osx.broker.eggroll.*;
import com.osx.broker.ptp.PtpForwardPushRespSO;
import com.osx.broker.router.RouterService;
import com.osx.broker.util.TransferUtil;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.*;
import com.osx.core.context.FateContext;
import com.osx.core.exceptions.*;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.ptp.SourceMethod;
import com.osx.core.ptp.TargetMethod;
import com.osx.core.utils.FlowLogUtil;
import com.osx.core.utils.ToStringUtils;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.command.Command;
import com.webank.eggroll.core.meta.Meta;
import com.webank.eggroll.core.transfer.Transfer;
import com.webank.eggroll.core.transfer.TransferServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.StringUtils;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
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
    FateContext context;
    ErRollSiteHeader rsHeader = null;
    TransferStatus transferStatus = TransferStatus.INIT;
    CountDownLatch finishLatch = new CountDownLatch(1);
    StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> putBatchSinkPushReqSO;
    RouterInfo routerInfo;
    Proxy.Metadata metadata;
    String brokerTag;
    private boolean isDst = false;
    private boolean needPrintFlow = true;
    private StreamObserver<Proxy.Packet> forwardPushReqSO;
    private StreamObserver<Proxy.Metadata> backRespSO;
    private Class  backRespSOClass;
    private String transferId;
    private Integer queueId;
    private RouterService  routerService;


    public QueuePushReqStreamObserver(Context context,RouterService routerService, StreamObserver backRespSO,
                                      Class backRespSOClass
    ) {
        this.context =(FateContext) context;
        this.routerService = routerService;
        this.backRespSOClass =  backRespSOClass;
        this.backRespSO = backRespSO;
        //this.context = context.subContext();
        //this.context.setNeedPrintFlowLog(true);
        this.context.setServiceName("pushTransfer");

    }

    public StreamObserver<Proxy.Packet> getForwardPushReqSO() {
        return forwardPushReqSO;
    }

    public void setForwardPushReqSO(StreamObserver<Proxy.Packet> forwardPushReqSO) {
        this.forwardPushReqSO = forwardPushReqSO;
    }

    public void init(Proxy.Packet packet) throws Exception {

        TransferUtil.assableContextFromProxyPacket(context,packet);
        Proxy.Metadata metadata = packet.getHeader();
        String desPartyId = context.getDesPartyId();
        String srcPartyId = context.getSrcPartyId();
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
            routerInfo =routerService.route(context.getSrcPartyId(),context.getSrcComponent(),context.getDesPartyId(),context.getDesComponent());
            if (routerInfo != null) {
                this.transferId = routerInfo.getResource();
            } else {
                throw new NoRouterInfoException("no router");
            }
        }
        if (isDst) {
            initEggroll(packet);
        } else {

            context.setActionType(ActionType.PUSH_REMOTE.getAlias());
            context.setRouterInfo(routerInfo);
            context.setSrcPartyId(routerInfo.getSourcePartyId());
            context.setDesPartyId(routerInfo.getDesPartyId());

            if (routerInfo.getProtocol().equals(Protocol.http))  {
                //由本方发起的传输且使用队列替代流式传输，需要在本地建立接受应答的队列,
                    forwardPushReqSO = QueueStreamBuilder.createStreamFromOrigin(context, backRespSO, Proxy.Packet.parser(),
                            routerInfo, srcPartyId, desPartyId, rsHeader.getRollSiteSessionId(),finishLatch);

            } else {
                ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(context.getRouterInfo(), true);
                if (TransferUtil.isOldVersionFate(routerInfo.getVersion())) {
                    DataTransferServiceGrpc.DataTransferServiceStub stub = DataTransferServiceGrpc.newStub(managedChannel);
                    ForwardPushRespSO forwardPushRespSO = new ForwardPushRespSO(context, backRespSO, backRespSOClass, () -> {
                        finishLatch.countDown();
                    }, (t) -> {
                        finishLatch.countDown();
                    });
                    forwardPushReqSO = stub.push(forwardPushRespSO);
                } else {
                        PtpForwardPushRespSO ptpForwardPushRespSO = new PtpForwardPushRespSO(context, backRespSO, backRespSOClass, () -> {
                            finishLatch.countDown();
                        }, (t) -> {
                            finishLatch.countDown();
                        });
                        PrivateTransferProtocolGrpc.PrivateTransferProtocolStub stub = PrivateTransferProtocolGrpc.newStub(managedChannel);
                        StreamObserver<Osx.Inbound> ptpForwardPushReqSO = stub.transport(ptpForwardPushRespSO);
                        forwardPushReqSO = new StreamObserver<Proxy.Packet>() {
                            @Override
                            public void onNext(Proxy.Packet packet) {
                                Osx.Inbound inbound = TransferUtil.buildInboundFromPushingPacket(packet, MetaInfo.PROPERTY_FATE_TECH_PROVIDER,TargetMethod.PUSH.name(), SourceMethod.PUSH.name()).build();
                                ptpForwardPushReqSO.onNext(inbound);
                            }

                            @Override
                            public void onError(Throwable throwable) {
                                ptpForwardPushReqSO.onError(throwable);
                            }

                            @Override
                            public void onCompleted() {
                                ptpForwardPushReqSO.onCompleted();
                            }
                        };
                }
            }
        }
        transferStatus = TransferStatus.TRANSFERING;
    }

    private void initEggroll(Proxy.Packet firstRequest) {
        if (StringUtils.isEmpty(MetaInfo.PROPERTY_EGGROLL_CLUSTER_MANANGER_IP)) {
            throw new SysException("eggroll cluter manager ip is not found");
        }

        metadata = firstRequest.getHeader();
        String oneLineStringMetadata = ToStringUtils.toOneLineString(metadata);
        context.setActionType(ActionType.PUSH_EGGROLL.getAlias());
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
            SessionInitException error = new SessionInitException("eggroll session "+sessionId+" invalid status : "+session.getErSessionMeta().getStatus());
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
                Lists.newArrayList(rp.getStore()),
                Lists.newArrayList(rp.getStore()),
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
            long begin = System.currentTimeMillis();
            try {
                Meta.Task taskMeta = Meta.Task.parseFrom(commandResponse.getResultsList().get(0));
                ErTask erTask = ErTask.parseFromPb(taskMeta);
                long now = System.currentTimeMillis();
                return erTask;
            } catch (InvalidProtocolBufferException igore) {

            }
            return null;
        });
        routerInfo = new RouterInfo();
        context.setRouterInfo(routerInfo);
        routerInfo.setHost(egg.getTransferEndpoint().getHost());
        routerInfo.setPort(egg.getTransferEndpoint().getPort());
        context.setSrcPartyId(routerInfo.getSourcePartyId());
        context.setDesPartyId(routerInfo.getDesPartyId());
        ManagedChannel channel = GrpcConnectionFactory.createManagedChannel(routerInfo,false);
        TransferServiceGrpc.TransferServiceStub stub = TransferServiceGrpc.newStub(channel);
        putBatchSinkPushReqSO = stub.send(new PutBatchSinkPushRespSO(metadata, commandFuture, backRespSO, finishLatch));
    }


    @Override
    public void onNext(Proxy.Packet value) {
        try {
            long seq = value.getHeader().getSeq();
            context.setDataSize(value.getSerializedSize());
            context.setCaseId(Long.toString(seq));
            if (transferStatus.equals(TransferStatus.INIT)) {
                init(value);
            }
            if (transferStatus.equals(TransferStatus.TRANSFERING)) {
                if (isDst) {
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

        logger.info("transferId {} receive completed", transferId);
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
                            needPrintFlow = false;
                        }
                    } catch (InterruptedException e) {
                        onError(e);
                        needPrintFlow = false;
                    }
                }

        }
    }

}
