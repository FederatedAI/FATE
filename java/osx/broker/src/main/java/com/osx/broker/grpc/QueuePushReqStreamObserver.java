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
import com.lmax.disruptor.EventHandler;
import com.osx.broker.ServiceContainer;
import com.osx.broker.constants.MessageFlag;
import com.osx.broker.consumer.EventDrivenConsumer;
import com.osx.broker.consumer.GrpcEventHandler;
import com.osx.broker.consumer.MessageEvent;
import com.osx.broker.eggroll.*;
import com.osx.broker.message.MessageExt;
import com.osx.broker.ptp.PtpForwardPushRespSO;
import com.osx.broker.queue.CreateQueueResult;
import com.osx.broker.queue.TransferQueue;
import com.osx.broker.util.TransferUtil;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.ActionType;
import com.osx.core.constant.Dict;
import com.osx.core.constant.StatusCode;
import com.osx.core.constant.TransferStatus;
import com.osx.core.context.Context;
import com.osx.core.exceptions.*;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.ptp.TargetMethod;
import com.osx.core.router.RouterInfo;
import com.osx.core.utils.FlowLogUtil;
import com.osx.core.utils.JsonUtil;
import com.osx.core.utils.ToStringUtils;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.command.Command;
import com.webank.eggroll.core.meta.Meta;
import com.webank.eggroll.core.transfer.Transfer;
import com.webank.eggroll.core.transfer.TransferServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.StringUtils;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class QueuePushReqStreamObserver implements StreamObserver<Proxy.Packet> {

    static public ConcurrentHashMap<Integer, QueuePushReqStreamObserver> queueIdMap = new ConcurrentHashMap<>();
    static AtomicInteger seq = new AtomicInteger(0);
    Logger logger = LoggerFactory.getLogger(QueuePushReqStreamObserver.class);
    Context context;
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


    public QueuePushReqStreamObserver(Context context, StreamObserver backRespSO,
                                      Class backRespSOClass
    ) {
        this.backRespSOClass =  backRespSOClass;
        this.backRespSO = backRespSO;
        this.context = context.subContext();
        this.context.setNeedPrintFlowLog(true);
        this.context.setServiceName("pushTransfer");
    }

    public StreamObserver<Proxy.Packet> getForwardPushReqSO() {
        return forwardPushReqSO;
    }

    public void setForwardPushReqSO(StreamObserver<Proxy.Packet> forwardPushReqSO) {
        this.forwardPushReqSO = forwardPushReqSO;
    }

    public void init(Proxy.Packet packet) throws Exception {

        Proxy.Metadata metadata = packet.getHeader();
        String desPartyId = metadata.getDst().getPartyId();
        String srcPartyId = metadata.getSrc().getPartyId();
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
            routerInfo = ServiceContainer.fateRouterService.route(packet);
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
            ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(context.getRouterInfo(), true);

            if (MetaInfo.PROPERTY_USE_MSG_QUEUE_REPLACE_STREAM && MetaInfo.PROPERTY_SELF_PARTY.contains(srcPartyId)) {
                //由本方发起的传输且使用队列替代流式传输，需要在本地建立接受应答的队列,
                PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub stub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
                String uuid = UUID.randomUUID().toString();
                String backTopic = Dict.EGGROLL_BACK_TOPIC_PREFIX + uuid;
                String sendTopic = Dict.EGGROLL_SEND_TOPIC_PREFIX + uuid;
                CreateQueueResult createQueueResult = ServiceContainer.transferQueueManager.createNewQueue(backTopic, context.getSessionId(), true);
                if (createQueueResult.getTransferQueue() == null) {
                    throw new RemoteRpcException("create queue error");
                }
                TransferQueue answerQueue = createQueueResult.getTransferQueue();
                ServiceContainer.consumerManager.createEventDrivenConsumer(backTopic, new GrpcEventHandler() {
                    @Override
                    protected void handleMessage(MessageExt message) {
                        Proxy.Metadata proxyMetadata = null;
                        try {
                            proxyMetadata = Proxy.Metadata.parseFrom(message.getBody());
                        } catch (InvalidProtocolBufferException e) {
                            e.printStackTrace();
                        }
                        backRespSO.onNext(proxyMetadata);
                    }

                    @Override
                    protected void handleError(MessageExt message) {
                        byte[] body = message.getBody();
                        ExceptionInfo exceptionInfo = JsonUtil.json2Object(body, ExceptionInfo.class);
                        backRespSO.onError(new RuntimeException(exceptionInfo.getMessage()));
                    }

                    @Override
                    protected void handleComplete(MessageExt message) {
                        backRespSO.onCompleted();
                    }
                });
                forwardPushReqSO = new StreamObserver<Proxy.Packet>() {
                    @Override
                    public void onNext(Proxy.Packet packet) {
                        Osx.Inbound inbound = TransferUtil.buildInboundFromPushingPacket(packet, TargetMethod.PRODUCE_MSG.name())
                                .putMetadata(Osx.Metadata.MessageTopicBack.name(), backTopic)
                                .putMetadata(Osx.Metadata.MessageFlag.name(), MessageFlag.SENDMSG.name())
                                .putMetadata(Osx.Metadata.MessageTopic.name(), sendTopic).build();
                        stub.invoke(inbound);
                    }

                    @Override
                    public void onError(Throwable throwable) {

                        ExceptionInfo  exceptionInfo = new ExceptionInfo();
                        exceptionInfo.setMessage(throwable.getMessage());
                        String errorData = JsonUtil.object2Json(exceptionInfo);
                        Osx.Inbound.Builder inboundBuilder = TransferUtil.buildInbound(srcPartyId,desPartyId,TargetMethod.PRODUCE_MSG.name(),
                                sendTopic,MessageFlag.ERROR,context.getSessionId(),errorData.getBytes(StandardCharsets.UTF_8))
                                .putMetadata(Osx.Metadata.MessageFlag.name(), MessageFlag.ERROR.name());
                        stub.invoke(inboundBuilder.build());
                    }

                    @Override
                    public void onCompleted() {
                        Osx.Inbound.Builder inboundBuilder = TransferUtil.buildInbound(srcPartyId,desPartyId,TargetMethod.PRODUCE_MSG.name(),
                                        sendTopic,MessageFlag.ERROR,context.getSessionId(),"completed".getBytes(StandardCharsets.UTF_8))
                                .putMetadata(Osx.Metadata.MessageFlag.name(), MessageFlag.COMPELETED.name());
                        stub.invoke(inboundBuilder.build());
                    }
                };

            } else {
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
                                Osx.Inbound inbound = TransferUtil.buildInboundFromPushingPacket(packet, TargetMethod.PUSH.name()).build();
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
            IllegalStateException error = new IllegalStateException("session=${sessionId} with illegal status. expected=${SessionStatus.ACTIVE}, actual=${session.sessionMeta.status}");
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
            logger.error("push error", e);
            ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
            context.setException(e);
            context.setReturnCode(exceptionInfo.getCode());
            throw new BaseException(exceptionInfo.getCode(), exceptionInfo.getMessage());
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
            //transferQueue.onError(t);


            putBatchSinkPushReqSO.onError(t);
        } else {

//            if(MetaInfo.PROPERTY_USE_QUEUE_MODEL){
//                if(transferQueue!=null){
//                    AbstractServiceAdaptor.ExceptionInfo   exceptionInfo  = new  AbstractServiceAdaptor.ExceptionInfo();
//                    exceptionInfo.setMessage(t.getMessage());
//                    exceptionInfo.setThrowable(t);
//                    MessageExtBrokerInner messageExtBrokerInner = MessageDecoder.buildMessageExtBrokerInner(transferId,exceptionInfo.toString().getBytes(StandardCharsets.UTF_8),
//                            queueId,MessageFlag.ERROR,routerInfo.getSourcePartyId(),routerInfo.getDesPartyId());
//                    transferQueue.putMessage(messageExtBrokerInner);
//                }
//            }else

            {
                if (forwardPushReqSO != null) {
                    forwardPushReqSO.onError(t);
                }
            }
        }


    }

    @Override
    public void onCompleted() {

        logger.info("transferId {} receive completed", transferId);
        if (isDst) {
//            if(transferQueue!=null) {
//                transferQueue.setWriteOver(true);
//            }
            if (putBatchSinkPushReqSO != null) {
                putBatchSinkPushReqSO.onCompleted();
            }
        } else {

            if (forwardPushReqSO != null) {

//                if(MetaInfo.PROPERTY_USE_QUEUE_MODEL){
//                    /**
//                     *  由pushConsumer去通知，因为要保证顺序，保证之前的数据传递完，所以只能放在队列最后串行执行
//                     */
//                    MessageExtBrokerInner messageExtBrokerInner = MessageDecoder.buildMessageExtBrokerInner(transferId,null,queueId,MessageFlag.COMPELETED,
//                            routerInfo.getSourcePartyId(),routerInfo.getDesPartyId());
//                    PutMessageResult putMessageResult = transferQueue.putMessage(messageExtBrokerInner);
//                }else

                {
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
            logger.info("receive completed  !!!!");
        }
    }

}
