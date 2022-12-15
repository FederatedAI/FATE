package com.osx.broker.ptp;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.osx.core.constant.Dict;
import com.osx.core.config.MetaInfo;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.router.RouterInfo;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.Context;
import com.osx.core.utils.ToStringUtils;
import com.osx.broker.eggroll.*;
import com.osx.broker.util.TransferUtil;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.command.Command;
import com.webank.eggroll.core.meta.Meta;
import com.webank.eggroll.core.transfer.Transfer;
import com.webank.eggroll.core.transfer.TransferServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.StringUtils;
import org.ppc.ptp.Pcp;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import static com.osx.broker.ServiceContainer.fateRouterService;

public class PtpPushReqStreamObserver  implements StreamObserver<Pcp.Inbound> {

    Logger logger = LoggerFactory.getLogger(PtpPushReqStreamObserver.class);
    RouterInfo routerInfo;
    boolean  inited = false;
    private boolean isDst=false;
    Context context;
    Proxy.Metadata metadata;
    String brokerTag;
    private  StreamObserver<Pcp.Outbound>  backRespSO;
    CountDownLatch finishLatch= new CountDownLatch(1);
    StreamObserver<Pcp.Inbound> ptpForwardPushReqSO;
    StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet>   oldPushReqSO;
    StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch>   putBatchSinkPushReqSO;

    private  void init(Pcp.Inbound inbound){
        Map<String,String> metaDataMap = inbound.getMetadataMap();
        logger.info("metaDataMap {}",metaDataMap);
        String version = metaDataMap.get(Pcp.Header.Version.name());
        String techProviderCode = metaDataMap.get(Pcp.Header.TechProviderCode.name());
        String traceId = metaDataMap.get(Pcp.Header.TraceID.name());
        String token = metaDataMap.get(Pcp.Header.Token.name());
        String sourceNodeId = metaDataMap.get(Pcp.Header.SourceNodeID.name());
        String targetNodeId = metaDataMap.get(Pcp.Header.TargetNodeID.name());
        String sourceInstId = metaDataMap.get(Pcp.Header.SourceInstID.name());
        String targetInstId = metaDataMap.get(Pcp.Header.TargetInstID.name());
        String sessionId = metaDataMap.get(Pcp.Header.SessionID.name());
        String targetMethod = metaDataMap.get(Pcp.Metadata.TargetMethod.name());
        String targetComponentName = metaDataMap.get(Pcp.Metadata.TargetComponentName.name());
        String sourceComponentName = metaDataMap.get(Pcp.Metadata.SourceComponentName.name());
        String sourcePartyId= StringUtils.isEmpty(sourceInstId)?sourceNodeId:sourceInstId+"."+sourceNodeId;
        String targetPartyId =StringUtils.isEmpty(targetInstId)? targetNodeId:targetInstId+"."+targetNodeId;
        String topic =  metaDataMap.get(Pcp.Metadata.MessageTopic.name());
        String offsetString = metaDataMap.get(Pcp.Metadata.MessageOffSet.name());
        routerInfo = fateRouterService.route(sourcePartyId,sourceComponentName,targetPartyId,targetComponentName);
        inited =true;
        if (MetaInfo.PROPERTY_SELF_PARTY.contains(targetNodeId)) {
            isDst = true;
        }
        if(isDst){
            Proxy.Packet  packet=null;
            try {
                  packet =  Proxy.Packet.parseFrom(inbound.getPayload());
            } catch (InvalidProtocolBufferException e) {
                e.printStackTrace();
            }
            if(packet!=null) {
                initEggroll(packet);
            }
        }else{
            ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo,null);
            if(TransferUtil.isOldVersionFate(routerInfo.getVersion())) {


                DataTransferServiceGrpc.DataTransferServiceStub stub = DataTransferServiceGrpc.newStub(managedChannel);
                oldPushReqSO = stub.push(new StreamObserver<Proxy.Metadata>() {
                    @Override
                    public void onNext(Proxy.Metadata metadata) {
                        Pcp.Outbound  outbound = Pcp.Outbound.newBuilder().setCode(StatusCode.SUCCESS).setMessage(Dict.SUCCESS).setPayload(metadata.toByteString()).build();
                        backRespSO.onNext(outbound);
                    }

                    @Override
                    public void onError(Throwable throwable) {
                        backRespSO.onError(throwable);
                    }
                    @Override
                    public void onCompleted() {
                        backRespSO.onCompleted();
                    }
                });
            }else{
                PtpForwardPushRespSO ptpForwardPushRespSO = new PtpForwardPushRespSO(context,backRespSO,"proxy" , null, null);
                PrivateTransferProtocolGrpc.PrivateTransferProtocolStub stub =PrivateTransferProtocolGrpc.newStub(managedChannel);
                ptpForwardPushReqSO =stub.transport(ptpForwardPushRespSO);

            }
        }


    }

    @Override
    public void onNext(Pcp.Inbound inbound) {
        if(!inited){
            init(inbound);
        }
        if(isDst){
            Proxy.Packet  packet = null;
            try {
                packet = Proxy.Packet.parseFrom(inbound.getPayload());
            } catch (InvalidProtocolBufferException e) {
                e.printStackTrace();
            }
            Transfer.TransferHeader.Builder  transferHeaderBuilder= Transfer.TransferHeader.newBuilder();
            Transfer.TransferHeader tbHeader = transferHeaderBuilder.setId((int)metadata.getSeq())
                    .setTag(brokerTag)
                    .setExt(packet.getHeader().getExt()).build();
            Transfer.TransferBatch.Builder  transferBatchBuilder  = Transfer.TransferBatch.newBuilder();
            Transfer.TransferBatch tbBatch = transferBatchBuilder.setHeader(tbHeader)
                    .setData(packet.getBody().getValue())
                    .build();
            putBatchSinkPushReqSO.onNext(tbBatch);
        }else {

            if (ptpForwardPushReqSO != null) {
                ptpForwardPushReqSO.onNext(inbound);
            } else if (oldPushReqSO != null) {
                Proxy.Packet packet = null;
                try {
                    packet = Proxy.Packet.parseFrom(inbound.getPayload());
                } catch (InvalidProtocolBufferException e) {
                    e.printStackTrace();
                }
                oldPushReqSO.onNext(packet);
            }
        }
    }

    @Override
    public void onError(Throwable throwable) {
        if(isDst){
            putBatchSinkPushReqSO.onError(throwable);
        }else{
            if (ptpForwardPushReqSO != null) {
                ptpForwardPushReqSO.onError(throwable);
            } else if(oldPushReqSO!=null){
                oldPushReqSO.onError(throwable);
            }
        }
    }

    @Override
    public void onCompleted() {
        if(isDst){
            putBatchSinkPushReqSO.onCompleted();
        }else{
            if (ptpForwardPushReqSO != null) {
                ptpForwardPushReqSO.onCompleted();
            } else if(oldPushReqSO!=null){
                oldPushReqSO.onCompleted();
            }
        }
    }


    private  void  initEggroll(Proxy.Packet  firstRequest){
        logger.info("init eggroll begin");
        metadata = firstRequest.getHeader();
        String oneLineStringMetadata = ToStringUtils.toOneLineString(metadata);
        ByteString encodedRollSiteHeader = metadata.getExt();
        context.setActionType("push-eggroll");
        ErRollSiteHeader rsHeader=null;
        try {
            rsHeader= ErRollSiteHeader.parseFromPb(Transfer.RollSiteHeader.parseFrom(encodedRollSiteHeader));
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();

        }
        //logger.info("=========ErRollSiteHeader {}",rsHeader);
        //"#", prefix: Array[String] = Array("__rsk")
        String rsKey = rsHeader.getRsKey("#","__rsk");
        String  sessionId = String.join("_", rsHeader.getRollSiteSessionId() , rsHeader.getDstRole(), rsHeader.getDstPartyId());
        context.setSessionId(sessionId);
        ErSession session = null;
        try {
            session = PutBatchSinkUtil.sessionCache.get(sessionId);
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        if (!SessionStatus.ACTIVE.name().equals(session.getErSessionMeta().getStatus())) {
            IllegalStateException error = new IllegalStateException("session=${sessionId} with illegal status. expected=${SessionStatus.ACTIVE}, actual=${session.sessionMeta.status}");
            onError(error);
            throw error;
        }

        String namespace = rsHeader.getRollSiteSessionId();
        String  name = rsKey;
        RollPairContext ctx = new RollPairContext(session);
        Map rpOptions = Maps.newHashMap();
        rpOptions.putAll(rsHeader.getOptions());
        rpOptions.put(Dict.TOTAL_PARTITIONS_SNAKECASE ,rsHeader.getTotalPartitions().toString());
        //var rpOptions = rsHeader.options ++ Map(StringConstants.TOTAL_PARTITIONS_SNAKECASE -> rsHeader.totalPartitions.toString)
        if (rsHeader.getDataType().equals("object")) {
            rpOptions.put(Dict.SERDES ,SerdesTypes.EMPTY.name());
        } else {
            rpOptions.put(Dict.SERDES ,rsHeader.getOptions().getOrDefault("serdes", SerdesTypes.PICKLE.name()));
        }

        // table creates here
        RollPair rp = ctx.load(namespace, name, rpOptions);

        Integer partitionId = rsHeader.getPartitionId();
        ErPartition partition = rp.getStore().getPartition(partitionId);
        ErProcessor egg = ctx.getErSession().routeToEgg(partition);
        logger.info("egg ========{}",egg);
//        RollPair.PUT_BATCH
//        delim: String = "#", prefix: Array[String] = Array("__rsk")
//        val PUT_BATCH = "putBatch"
        // s"${}-${rsHeader.getRsKey()}-${partitionId}"
        brokerTag =
                "putBatch-"+rsHeader.getRsKey("#", "__rsk")+"-"+partitionId;
        logger.info("======= brokerTag ======{}",brokerTag);
        String  jobId = IdUtils.generateJobId(ctx.getErSession().getSessionId(),  brokerTag,"-");
        logger.info("======jobId ======={}",jobId);
        Map<String,String > jobOptions = new HashMap<>();

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
            logger.info("before  call EGG_RUN_TASK_COMMAND  {}",task);
            Command.CommandResponse commandResponse  = commandClient.call(RollPair.EGG_RUN_TASK_COMMAND, task);
            logger.info("============== EGG_RUN_TASK_COMMAND {}",commandResponse);

            long  begin = System.currentTimeMillis();
            try {
                Meta.Task taskMeta =   Meta.Task.parseFrom(commandResponse.getResultsList().get(0));
                ErTask  erTask =  ErTask.parseFromPb(taskMeta);
                long  now =  System.currentTimeMillis();
                logger.info("task ===cost===={}",now -begin);

                return   erTask;
            } catch (InvalidProtocolBufferException e) {
                e.printStackTrace();
            }
            return null;
        });
        routerInfo = new RouterInfo();
        context.setRouterInfo(routerInfo);
        routerInfo.setHost(egg.getTransferEndpoint().getHost());
        routerInfo.setPort(egg.getTransferEndpoint().getPort());
        context.setSrcPartyId(routerInfo.getSourcePartyId());
        context.setDesPartyId(routerInfo.getDesPartyId());
        ManagedChannel channel = GrpcConnectionFactory.createManagedChannel(routerInfo);
        TransferServiceGrpc.TransferServiceStub stub = TransferServiceGrpc.newStub(channel);
        putBatchSinkPushReqSO = stub.send(new PutBatchSinkPushRespSO(metadata, commandFuture, new StreamObserver<Proxy.Metadata>() {
            @Override
            public void onNext(Proxy.Metadata metadata) {
                /**
                 * 这里需要转换
                 */
                Pcp.Outbound.Builder  outboundBuilder = Pcp.Outbound.newBuilder();
                outboundBuilder.setCode(StatusCode.SUCCESS);
                outboundBuilder.setMessage(Dict.SUCCESS);
                outboundBuilder.setPayload(metadata.toByteString());
                Pcp.Outbound  outbound = outboundBuilder.build();
                backRespSO.onNext(outbound);
            }

            @Override
            public void onError(Throwable throwable) {
                backRespSO.onError(throwable);
            }

            @Override
            public void onCompleted() {
                backRespSO.onCompleted();
            }
        }, finishLatch));

        logger.info("eggroll init over =========");
    }
}
