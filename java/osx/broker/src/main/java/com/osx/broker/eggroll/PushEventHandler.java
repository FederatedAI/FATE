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
import com.osx.broker.constants.MessageFlag;
import com.osx.broker.consumer.EventDrivenConsumer;
import com.osx.broker.consumer.GrpcEventHandler;
import com.osx.broker.consumer.MessageEvent;
import com.osx.broker.consumer.UnaryConsumer;
import com.osx.broker.message.MessageExt;
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
import com.osx.core.frame.Lifecycle;
import com.osx.core.ptp.TargetMethod;
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
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

public class PushEventHandler extends GrpcEventHandler {


    Logger logger = LoggerFactory.getLogger(PushEventHandler.class);
    TransferStatus  transferStatus= TransferStatus.INIT;
    Context context = new Context();
    RouterInfo routerInfo ;
    Proxy.Metadata metadata;
    String brokerTag;
    ErRollSiteHeader rsHeader = null;
    CountDownLatch finishLatch;
    StreamObserver<Transfer.TransferBatch> putBatchSinkPushReqSO;
    String  topic = null;



    protected   void handleError(MessageExt messageExt){
        //需要构建新异常
        putBatchSinkPushReqSO.onError(new Exception());
    }

    protected  void handleComplete(MessageExt messageExt){

        putBatchSinkPushReqSO.onCompleted();
    }

    protected  void  handleMessage(MessageExt messageExt){
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

            TransferStatus transferStatus = TransferStatus.INIT;

            RouterInfo revertRouterInfo;
            PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub blockingStub;
            String backTopic = null;
            String srcPartyId = context.getSrcPartyId();
            String desPartyId = context.getDesPartyId();
            String sessionId =  context.getSessionId();

            private void init(){
                //String srcPartyId, String srcRole, String dstPartyId, String desRole
                RouterInfo revertRouterInfo = ServiceContainer.fateRouterService.route(desPartyId,"",srcPartyId,"");
                if(revertRouterInfo==null){
                    throw new NoRouterInfoException(srcPartyId+" can not found route info");
                }
                ManagedChannel channel = GrpcConnectionFactory.createManagedChannel(routerInfo,true);
                blockingStub = PrivateTransferProtocolGrpc.newBlockingStub(channel);
                backTopic = buildBackTopic(topic);
                transferStatus= TransferStatus.TRANSFERING;
            }

            @Override
            public void onNext(Proxy.Metadata metadata) {
                /**
                 * 数据回调
                 */
                if(transferStatus.equals(TransferStatus.INIT)){
                    init();
                }
                if(transferStatus.equals(TransferStatus.TRANSFERING)){
                    //将其对调后再查路由
                    Osx.Inbound.Builder  inboundBuilder = TransferUtil.buildInbound(desPartyId,srcPartyId,TargetMethod.PRODUCE_MSG.name(),
                            backTopic,MessageFlag.SENDMSG,sessionId, metadata.toByteString().toByteArray());
                    Osx.Outbound outbound = blockingStub.invoke(inboundBuilder.build());
                }
            }

            @Override
            public void onError(Throwable throwable) {
                ExceptionInfo  exceptionInfo = new ExceptionInfo();
                exceptionInfo.setMessage(throwable.getMessage());
                String message = throwable.getMessage();
                Osx.Inbound.Builder  inboundBuilder = TransferUtil.buildInbound(desPartyId,srcPartyId,TargetMethod.PRODUCE_MSG.name(),
                        backTopic,MessageFlag.SENDMSG,sessionId,exceptionInfo.toString().getBytes(StandardCharsets.UTF_8) );
                Osx.Outbound outbound = blockingStub.invoke(inboundBuilder.build());
            }

            @Override
            public void onCompleted() {
                /**
                 * 完成回调
                 */
                Osx.Inbound.Builder  inboundBuilder = TransferUtil.buildInbound(desPartyId,srcPartyId,TargetMethod.PRODUCE_MSG.name(),
                        backTopic,MessageFlag.SENDMSG,sessionId,"completed".getBytes(StandardCharsets.UTF_8) );
                Osx.Outbound outbound = blockingStub.invoke(inboundBuilder.build());

            }
        }, finishLatch));




        transferStatus= TransferStatus.TRANSFERING;
    }

    private  String  buildBackTopic(String oriTopic){
        int length = Dict.EGGROLL_SEND_TOPIC_PREFIX.length();
        return Dict.EGGROLL_BACK_TOPIC_PREFIX+oriTopic.substring(length);
    }




}
