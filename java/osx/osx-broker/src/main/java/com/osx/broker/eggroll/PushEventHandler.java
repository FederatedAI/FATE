package com.osx.broker.eggroll;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.osx.api.constants.Protocol;
import com.osx.api.router.RouterInfo;
import com.osx.broker.ServiceContainer;
import com.osx.broker.constants.MessageFlag;
import com.osx.broker.consumer.GrpcEventHandler;
import com.osx.broker.consumer.MessageEvent;
import com.osx.broker.message.MessageExt;
import com.osx.broker.util.TransferUtil;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.*;
import com.osx.core.context.FateContext;
import com.osx.core.exceptions.*;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.ptp.TargetMethod;
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
    public PushEventHandler(){
        super(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
    }
    TransferStatus  transferStatus= TransferStatus.INIT;
    FateContext context = new FateContext();
    RouterInfo routerInfo ;
    Proxy.Metadata metadata;
    String brokerTag;
    ErRollSiteHeader rsHeader = null;
    CountDownLatch finishLatch;
    StreamObserver<Transfer.TransferBatch> putBatchSinkPushReqSO;
    String  topic = null;
    String backTopic = null;
    PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub backBlockingStub;
    String desRole = null;
    String srcRole = null;
    String sessionId = null;
    RouterInfo revertRouterInfo;

    protected   void handleError(MessageExt messageExt){
        //todo
        // 需要构建新异常
        try {

            if (putBatchSinkPushReqSO != null) {
                putBatchSinkPushReqSO.onError(new Exception());
            }
        }finally {
            String topic =  messageExt.getTopic();
            ServiceContainer.transferQueueManager.onCompleted(topic);
        }
    }

    protected  void handleComplete(MessageExt messageExt){
        try {
            if (putBatchSinkPushReqSO != null) {
                putBatchSinkPushReqSO.onCompleted();
            }
        }finally {
           String topic =  messageExt.getTopic();
           ServiceContainer.transferQueueManager.onCompleted(topic);
        }


    }

    @Override
    protected void handleInit(MessageExt event) {

    }

    protected  void  handleMessage(MessageExt messageExt){
        try {
            Proxy.Packet packet=null;
            try {
                packet = Proxy.Packet.parseFrom(messageExt.getBody());
            }catch (Exception  e){
                logger.error("parse packet error {}",new String(messageExt.getBody()));
            }
            if (transferStatus.equals(TransferStatus.INIT)) {
                //初始化
                try {
                    initEggroll(packet,messageExt);
                }catch(Exception e){
                    logger.error("init eggroll error",e);
                    transferStatus=TransferStatus.ERROR;
                }
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
            logger.error("handle msg error : "+ messageExt.getTopic(),e);
            if(backBlockingStub!=null) {
                Osx.Inbound.Builder inboundBuilder = TransferUtil.buildInbound(provider,desPartyId, srcPartyId, TargetMethod.PRODUCE_MSG.name(),
                        backTopic, MessageFlag.ERROR, sessionId, ErrorMessageUtil.buildRemoteRpcErrorMsg(1343,"kkkkk").getBytes());
                Osx.Outbound outbound = backBlockingStub.invoke(inboundBuilder.build());
            }else{
                logger.error("back stub is null");
            }
        }
    }

    private void initEggroll(Proxy.Packet firstRequest,MessageExt messageExt) throws Exception {
        if (StringUtils.isEmpty(MetaInfo.PROPERTY_EGGROLL_CLUSTER_MANANGER_IP)) {
            throw new SysException("eggroll cluter manager ip is not found");
        }

        topic = messageExt.getTopic();
        backTopic= buildBackTopic(topic);
        metadata = firstRequest.getHeader();
        ByteString encodedRollSiteHeader = metadata.getExt();
        rsHeader = ErRollSiteHeader.parseFromPb(Transfer.RollSiteHeader.parseFrom(encodedRollSiteHeader));
        Integer partitionId = rsHeader.getPartitionId();
        brokerTag = "putBatch-" + rsHeader.getRsKey("#", "__rsk") + "-" + partitionId;
        String oneLineStringMetadata = ToStringUtils.toOneLineString(metadata);
        context.setActionType(ActionType.PUSH_EGGROLL.getAlias());
        String rsKey = rsHeader.getRsKey("#", "__rsk");
        sessionId = String.join("_", rsHeader.getRollSiteSessionId(), rsHeader.getDstRole(), rsHeader.getDstPartyId());
        context.setSessionId(sessionId);
        desPartyId = metadata.getDst().getPartyId();
        desRole = metadata.getDst().getRole();
        srcRole = metadata.getSrc().getRole();
        srcPartyId = metadata.getSrc().getPartyId();
        //String srcPartyId, String srcRole, String dstPartyId, String desRole
        revertRouterInfo = ServiceContainer.routerRegister.getRouterService(MetaInfo.PROPERTY_FATE_TECH_PROVIDER).route(desPartyId,desRole,srcPartyId,srcRole);
        if(revertRouterInfo==null){
            throw new NoRouterInfoException(srcPartyId+" can not found route info");
        }
        if(Protocol.grpc.equals(revertRouterInfo.getProtocol())) {
            ManagedChannel backChannel = GrpcConnectionFactory.createManagedChannel(revertRouterInfo, true);
            backBlockingStub = PrivateTransferProtocolGrpc.newBlockingStub(backChannel);
            context.putData(Dict.BLOCKING_STUB,backBlockingStub);
        }


        ErSession session = null;
        try {
            session = PutBatchSinkUtil.sessionCache.get(sessionId);
        } catch (ExecutionException e) {
            logger.error("get session error ", e);
        }
        if (!SessionStatus.ACTIVE.name().equals(session.getErSessionMeta().getStatus())) {
            logger.error("");
            IllegalStateException error = new IllegalStateException("eggroll  session "+sessionId+" status is "+session.getErSessionMeta().getStatus());
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
        ManagedChannel eggChannel = GrpcConnectionFactory.createManagedChannel(routerInfo,false);
        TransferServiceGrpc.TransferServiceStub stub = TransferServiceGrpc.newStub(eggChannel);
        StreamObserver<Proxy.Metadata> eggSiteServicerPushRespSO;
        putBatchSinkPushReqSO = stub.send(new PutBatchSinkPushRespSO(metadata, commandFuture, new  StreamObserver<Proxy.Metadata>(){

            TransferStatus transferStatus = TransferStatus.INIT;

            private void init(){
                transferStatus= TransferStatus.TRANSFERING;
            }

            @Override
            public void onNext(Proxy.Metadata metadata) {
                    //将其对调后再查路由
                    Osx.Inbound.Builder  inboundBuilder = TransferUtil.buildInbound(provider,desPartyId,srcPartyId,TargetMethod.PRODUCE_MSG.name(),
                            backTopic,MessageFlag.SENDMSG,sessionId, metadata.toByteString().toByteArray());
                    TransferUtil.redirect(context,inboundBuilder.build(),revertRouterInfo,true);
            }

            @Override
            public void onError(Throwable throwable) {
                    ExceptionInfo exceptionInfo = new ExceptionInfo();
                    exceptionInfo.setMessage(throwable.getMessage());
                    String message = throwable.getMessage();
                    Osx.Inbound.Builder inboundBuilder = TransferUtil.buildInbound(provider,desPartyId, srcPartyId, TargetMethod.PRODUCE_MSG.name(),
                            backTopic, MessageFlag.SENDMSG, sessionId, exceptionInfo.toString().getBytes(StandardCharsets.UTF_8));
                    TransferUtil.redirect(context,inboundBuilder.build(),revertRouterInfo,true);

            }

            @Override
            public void onCompleted() {
                /**
                 * 完成回调
                 */
                try {
                        Osx.Inbound.Builder inboundBuilder = TransferUtil.buildInbound(provider,desPartyId, srcPartyId, TargetMethod.PRODUCE_MSG.name(),
                                backTopic, MessageFlag.COMPELETED, sessionId, "completed".getBytes(StandardCharsets.UTF_8));
                        Osx.Outbound result =TransferUtil.redirect(context, inboundBuilder.build(), revertRouterInfo,true);
                }catch (Exception e){
                    logger.error("receive completed error",e);
                }
            }
        }, finishLatch));
        transferStatus= TransferStatus.TRANSFERING;
    }

    private  String  buildBackTopic(String oriTopic){
        int length = Dict.STREAM_SEND_TOPIC_PREFIX.length();
        return Dict.STREAM_BACK_TOPIC_PREFIX+oriTopic.substring(length);
    }
}
