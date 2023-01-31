package com.osx.broker.eggroll;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.protobuf.InvalidProtocolBufferException;
import com.lmax.disruptor.BlockingWaitStrategy;
import com.lmax.disruptor.EventHandler;
import com.lmax.disruptor.dsl.Disruptor;
import com.lmax.disruptor.dsl.ProducerType;
import com.lmax.disruptor.util.DaemonThreadFactory;
import com.osx.broker.ServiceContainer;
import com.osx.broker.consumer.UnaryConsumer;
import com.osx.broker.message.MessageExt;
import com.osx.broker.queue.TransferQueue;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.ActionType;
import com.osx.core.constant.Dict;
import com.osx.core.constant.StatusCode;
import com.osx.core.constant.TransferStatus;
import com.osx.core.context.Context;
import com.osx.core.exceptions.*;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.frame.Lifecycle;
import com.osx.core.router.RouterInfo;
import com.osx.core.utils.ToStringUtils;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.command.Command;
import com.webank.eggroll.core.meta.Meta;
import com.webank.eggroll.core.transfer.Transfer;
import com.webank.eggroll.core.transfer.TransferServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

public class PushEventHandler implements EventHandler<MessageEvent> {


    Logger logger = LoggerFactory.getLogger(PushEventHandler.class);
    TransferStatus  transferStatus= TransferStatus.INIT;

    String topic;
    Context context = new Context();
    RouterInfo routerInfo ;
    Proxy.Metadata metadata;
    String brokerTag;
    ErRollSiteHeader rsHeader = null;
    CountDownLatch finishLatch;
    StreamObserver<Transfer.TransferBatch> putBatchSinkPushReqSO;

    @Override
    public void onEvent(MessageEvent streamMessageEvent, long l, boolean b) throws Exception {
        UnaryConsumer consumer = ServiceContainer.consumerManager.getOrCreateUnaryConsumer(topic);
        TransferQueue.TransferQueueConsumeResult transferQueueConsumeResult = consumer.consume(context, -1);
        if (transferQueueConsumeResult.getCode().equals(StatusCode.SUCCESS)) {

            MessageExt messageExt = transferQueueConsumeResult.getMessage();
            int flag = messageExt.getFlag();
            switch (flag){
                //msg
                case 0:  handleMsg(messageExt);break;
                //error
                case 1: handleError(messageExt);break;
                //completed
                case 2: handleComplete(messageExt);break;
            }

        }else{
            logger.warn("");
        }

    }

    private  void handleError(MessageExt messageExt){
        //需要构建新异常
        putBatchSinkPushReqSO.onError(new Exception());
    }

    private  void handleComplete(MessageExt messageExt){

        putBatchSinkPushReqSO.onCompleted();
    }

    private  void  handleMsg(MessageExt messageExt){
        try {
            Proxy.Packet packet = Proxy.Packet.parseFrom(messageExt.getBody());
            if (transferStatus.equals(TransferStatus.INIT)) {
                //初始化
                initEggroll(packet);

            }
            if (!transferStatus.equals(TransferStatus.TRANSFERING)) {
                throw new RemoteRpcException("eggroll init error");
            }

            Transfer.TransferHeader.Builder transferHeaderBuilder = Transfer.TransferHeader.newBuilder();
            Transfer.TransferHeader tbHeader = transferHeaderBuilder.setId((int) metadata.getSeq())
                    .setTag(brokerTag)
                    .setExt(packet.getHeader().getExt()).build();
            Transfer.TransferBatch.Builder transferBatchBuilder = Transfer.TransferBatch.newBuilder();
            Transfer.TransferBatch tbBatch = transferBatchBuilder.setHeader(tbHeader)
                    .setData(packet.getBody().getValue())
                    .build();
            putBatchSinkPushReqSO.onNext(tbBatch);
        }catch(Exception e){
            e.printStackTrace();
        }
    }


//    @Override
//    public void onNext(Proxy.Packet value) {
//        try {
//            long seq = value.getHeader().getSeq();
//            context.setDataSize(value.getSerializedSize());
//            context.setCaseId(Long.toString(seq));
//            if (transferStatus.equals(TransferStatus.INIT)) {
//                init(value);
//            }
//            if (transferStatus.equals(TransferStatus.TRANSFERING)) {
////                if (isDst) {
//                    Transfer.TransferHeader.Builder transferHeaderBuilder = Transfer.TransferHeader.newBuilder();
//                    Transfer.TransferHeader tbHeader = transferHeaderBuilder.setId((int) metadata.getSeq())
//                            .setTag(brokerTag)
//                            .setExt(value.getHeader().getExt()).build();
//                    Transfer.TransferBatch.Builder transferBatchBuilder = Transfer.TransferBatch.newBuilder();
//                    Transfer.TransferBatch tbBatch = transferBatchBuilder.setHeader(tbHeader)
//                            .setData(value.getBody().getValue())
//                            .build();
//                    putBatchSinkPushReqSO.onNext(tbBatch);
////                } else {
////                    forwardPushReqSO.onNext(value);
////                }
//            }
//
//
//        } catch (Exception e) {
//            logger.error("push error", e);
//            ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
//            context.setException(e);
//            context.setReturnCode(exceptionInfo.getCode());
//            throw new BaseException(exceptionInfo.getCode(), exceptionInfo.getMessage());
//        } finally {
//            FlowLogUtil.printFlowLog(context);
//        }
//    }



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
        //    onError(error);
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
        StreamObserver<Proxy.Metadata> eggSiteServicerPushRespSO;
        putBatchSinkPushReqSO = stub.send(new PutBatchSinkPushRespSO(metadata, commandFuture, new  StreamObserver<Proxy.Metadata>(){

            @Override
            public void onNext(Proxy.Metadata metadata) {
                /**
                 * 数据回调
                 */




            }

            @Override
            public void onError(Throwable throwable) {
                /**
                 * 异常回调
                 */
            }

            @Override
            public void onCompleted() {
                /**
                 * 完成回调
                 */
            }
        }, finishLatch));


        transferStatus= TransferStatus.TRANSFERING;
    }




}
