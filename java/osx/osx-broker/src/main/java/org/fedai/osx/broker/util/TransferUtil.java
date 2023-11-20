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
package org.fedai.osx.broker.util;


import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import org.apache.commons.lang3.StringUtils;


import org.fedai.osx.broker.constants.MessageFlag;
import org.fedai.osx.broker.http.HttpClientPool;
import org.fedai.osx.broker.http.HttpDataWrapper;
import org.fedai.osx.broker.http.HttpsClientPool;
import org.fedai.osx.broker.pojo.HttpInvoke;
import org.fedai.osx.broker.pojo.HttpInvokeResult;
import org.fedai.osx.broker.queue.TransferQueueConsumeResult;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.config.TransferMeta;
import org.fedai.osx.core.constant.*;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.context.Protocol;
import org.fedai.osx.core.exceptions.*;
import org.fedai.osx.core.frame.GrpcConnectionFactory;
import org.fedai.osx.core.ptp.SourceMethod;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.utils.AssertUtil;
import org.fedai.osx.core.utils.JsonUtil;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.ppc.ptp.PrivateTransferTransportGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.management.MBeanServer;
import javax.servlet.ServletResponse;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.util.Map;

public class TransferUtil {


    static Logger logger = LoggerFactory.getLogger(TransferUtil.class);


    /**
     * 2.0之前版本
     *
     * @param version
     * @return
     */
    public static boolean isOldVersionFate(String version) {

        try {
            if (StringUtils.isEmpty(version))
                version = MetaInfo.PROPERTY_DEFAULT_CLIENT_VERSION;
            String firstVersion = version.substring(0, 1);
            if (Integer.parseInt(firstVersion) >= 2) {
                return false;
            } else {
                return true;
            }
        } catch (NumberFormatException e) {
            throw new ConfigErrorException("remote version config error : " + version);
        }

    }


    public static String buildResource(Osx.Inbound inbound) {
        String sourceNodeId = inbound.getMetadataMap().get(Osx.Header.SourceNodeID.name());
        String targetNodeId = inbound.getMetadataMap().get(Osx.Header.TargetNodeID.name());
        String sourceInstId = inbound.getMetadataMap().get(Osx.Header.SourceInstID.name());
        if (sourceInstId == null) {
            sourceInstId = "";
        }
        String targetInstId = inbound.getMetadataMap().get(Osx.Header.TargetInstID.name());
        if (targetInstId == null) {
            targetInstId = "";
        }
        StringBuffer sb = new StringBuffer();
        sb.append(sourceInstId).append(sourceNodeId).append("_").append(targetInstId).append(targetNodeId);
        return sb.toString();
    }

    public static Proxy.Metadata buildProxyMetadataFromOutbound(Osx.Outbound outbound) {
        try {
            return Proxy.Metadata.parseFrom(outbound.getPayload());
        } catch (InvalidProtocolBufferException e) {

        }
        return null;
    }

    public static Osx.Outbound buildOutboundFromProxyMetadata(Proxy.Metadata metadata) {
        return Osx.Outbound.newBuilder().setPayload(metadata.toByteString()).build();

    }

    public static Proxy.Packet parsePacketFromInbound(Osx.Inbound inbound) {
        try {
            return Proxy.Packet.parseFrom(inbound.getPayload());
        } catch (InvalidProtocolBufferException e) {
            return null;
        }
    }

    public static Osx.Inbound.Builder buildInbound(String provider,
                                                   String srcPartyId,
                                                   String desPartyId,
                                                   String targetMethod,
                                                   String topic,
                                                   MessageFlag messageFlag,
                                                   String sessionId,
                                                   byte[] payLoad) {

        Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();

        if (payLoad != null) {
            inboundBuilder.setPayload(ByteString.copyFrom(payLoad));
        }
        return inboundBuilder;

    }


    public static TransferMeta parseTransferMetaFromProxyPacket(Proxy.Packet packet) {
        TransferMeta transferMeta = new TransferMeta();

//        logger.info("========package {}",packet);
        Proxy.Metadata metadata = packet.getHeader();
        Transfer.RollSiteHeader rollSiteHeader = null;

//        logger.info("========package header {} ext {}",metadata,metadata.getExt().size());

        String dstPartyId = null;
        String srcPartyId = null;
        String desRole = null;
        String srcRole = null;
        try {
            rollSiteHeader = Transfer.RollSiteHeader.parseFrom(metadata.getExt());
//            logger.info("========rollsite header {}",rollSiteHeader);
        } catch (InvalidProtocolBufferException e) {
            throw new ParameterException("invalid rollSiteHeader");
        }
        String sessionId = "";
        if (rollSiteHeader != null) {
            dstPartyId = rollSiteHeader.getDstPartyId();
            srcPartyId = rollSiteHeader.getSrcPartyId();
            desRole = rollSiteHeader.getDstRole();
            srcRole = rollSiteHeader.getSrcRole();
        }
        if (StringUtils.isEmpty(dstPartyId)) {
            dstPartyId = metadata.getDst().getPartyId();
        }
        if (StringUtils.isEmpty(desRole)) {
            desRole = metadata.getDst().getRole();
        }
        if (StringUtils.isEmpty(srcRole)) {
            srcRole = metadata.getSrc().getRole();
        }
        if (StringUtils.isEmpty(srcPartyId)) {
            srcPartyId = metadata.getSrc().getPartyId();
        }

        if (rollSiteHeader != null) {
            sessionId = String.join("_", rollSiteHeader.getRollSiteSessionId(), desRole, dstPartyId);
        }
        if (metadata.getDst() != null) {
            transferMeta.setTopic(metadata.getDst().getName());
        }

        transferMeta.setDesPartyId(dstPartyId);
        transferMeta.setSrcPartyId(srcPartyId);
        transferMeta.setDesRole(desRole);
        transferMeta.setSrcRole(srcRole);
        transferMeta.setSessionId(sessionId);
        return transferMeta;
    }



    public static void assableContextFromProxyPacket(OsxContext context, Proxy.Packet packet) {
        TransferMeta transferMeta = parseTransferMetaFromProxyPacket(packet);
       // logger.info("========  assableContextFromProxyPacket {}",transferMeta);
        context.setSrcNodeId(transferMeta.getSrcPartyId());
        context.setDesNodeId(transferMeta.getDesPartyId());
        context.setSrcComponent(transferMeta.getSrcRole());
        context.setDesComponent(transferMeta.getDesRole());
        context.setSessionId(transferMeta.getSessionId());
        context.setTopic(transferMeta.getTopic());
        context.setTechProviderCode(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        if (MetaInfo.PROPERTY_SELF_PARTY.contains(context.getDesNodeId())) {
            context.setSelfPartyId(context.getDesNodeId());
        } else {
            context.setSelfPartyId(MetaInfo.PROPERTY_SELF_PARTY.toArray()[0].toString());
        }

    }


    public static Osx.Inbound.Builder buildInboundFromPushingPacket(OsxContext  context ,Proxy.Packet packet, String provider) {
        Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
        TransferMeta transferMeta = parseTransferMetaFromProxyPacket(packet);
        inboundBuilder.setPayload(packet.toByteString());
//        inboundBuilder.putMetadata(Osx.Header.Version.name(), MetaInfo.CURRENT_VERSION);
//        inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(), provider);
//        inboundBuilder.putMetadata(Osx.Header.Token.name(), "");
//        inboundBuilder.putMetadata(Osx.Header.SourceNodeID.name(), transferMeta.getSrcPartyId());
//        inboundBuilder.putMetadata(Osx.Header.TargetNodeID.name(), transferMeta.getDesPartyId());
//        inboundBuilder.putMetadata(Osx.Header.SourceInstID.name(), "");
//        inboundBuilder.putMetadata(Osx.Header.TargetInstID.name(), "");
//        inboundBuilder.putMetadata(Osx.Metadata.SourceMethod.name(), sourceMethod);
//        inboundBuilder.putMetadata(Osx.Header.SessionID.name(), transferMeta.getSessionId());
//        inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), targetMethod);
//        inboundBuilder.putMetadata(Osx.Metadata.TargetComponentName.name(), transferMeta.getDesRole());
//        inboundBuilder.putMetadata(Osx.Metadata.SourceComponentName.name(), "");
        return inboundBuilder;

    }

    ;


    static public Osx.Inbound.Builder buildPbFromHttpRequest(OsxContext context, HttpServletRequest request) {

        Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
        String version = request.getHeader(PtpHttpHeader.Version);
        String techProviderCode = request.getHeader(PtpHttpHeader.TechProviderCode);
        String traceID = request.getHeader(PtpHttpHeader.TraceID);
        String token = request.getHeader(PtpHttpHeader.Token);
        String sourceNodeID = request.getHeader(PtpHttpHeader.FromNodeID);
        String targetNodeID = request.getHeader(PtpHttpHeader.TargetNodeID);
        String sourceInstID = request.getHeader(PtpHttpHeader.FromInstID);
        String targetInstID = request.getHeader(PtpHttpHeader.TargetInstID);
        String sessionID = request.getHeader(PtpHttpHeader.SessionID);
        String uri =   request.getHeader(PtpHttpHeader.Uri);

        String messageTopic = request.getHeader(PtpHttpHeader.MessageTopic);
        String messageCode = request.getHeader(Osx.Metadata.MessageCode.name());
        String retryCount = request.getHeader(Osx.Metadata.RetryCount.name());
        String sourceComponentName = request.getHeader(PtpHttpHeader.SourceComponentName);
        String targetComponentName = request.getHeader(PtpHttpHeader.TargetComponentName);
        String messageOffSet = request.getHeader(PtpHttpHeader.MessageOffSet);
        String instanceId = request.getHeader(PtpHttpHeader.InstanceId);
        String timestamp = request.getHeader(PtpHttpHeader.Timestamp);
        String messageFlag = request.getHeader(PtpHttpHeader.MessageFlag);
        String jobId = request.getHeader(PtpHttpHeader.JobId);


        context.setSrcNodeId(sourceNodeID);
        context.setDesNodeId(targetNodeID);
        context.setSessionId(sessionID);
        context.setTopic(messageTopic);
//        context.setActionType(targetMethod);
        context.setUri(uri);
        context.setProtocol(Protocol.http);
//        inboundBuilder.putMetadata(Osx.Header.Version.name(), version != null ? version : "");
//        inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(), techProviderCode != null ? techProviderCode : "");
//        inboundBuilder.putMetadata(Osx.Header.Token.name(), token != null ? token : "");
//        inboundBuilder.putMetadata(Osx.Header.SourceNodeID.name(), sourceNodeID != null ? sourceNodeID : "");
//        inboundBuilder.putMetadata(Osx.Header.TargetNodeID.name(), targetNodeID != null ? targetNodeID : "");
//        inboundBuilder.putMetadata(Osx.Header.SourceInstID.name(), sourceInstID != null ? sourceInstID : "");
//        inboundBuilder.putMetadata(Osx.Header.TargetInstID.name(), targetInstID != null ? targetInstID : "");
//        inboundBuilder.putMetadata(Osx.Header.SessionID.name(), sessionID != null ? sessionID : "");
////        inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), targetMethod != null ? targetMethod : "");
//        inboundBuilder.putMetadata(Osx.Metadata.TargetComponentName.name(), targetComponentName != null ? targetComponentName : "");
//        inboundBuilder.putMetadata(Osx.Metadata.SourceComponentName.name(), sourceComponentName != null ? sourceComponentName : "");
//        inboundBuilder.putMetadata(Osx.Metadata.MessageTopic.name(), messageTopic != null ? messageTopic : "");
//        inboundBuilder.putMetadata(Osx.Metadata.MessageOffSet.name(), messageOffSet != null ? messageOffSet : "");
//        inboundBuilder.putMetadata(Osx.Metadata.InstanceId.name(), instanceId != null ? instanceId : "");
//        inboundBuilder.putMetadata(Osx.Metadata.Timestamp.name(), timestamp != null ? timestamp : "");
//        inboundBuilder.putMetadata(Osx.Metadata.SourceMethod.name(), sourceMethod != null ? sourceMethod : "");
//        inboundBuilder.putMetadata(Osx.Metadata.MessageFlag.name(), messageFlag != null ? messageFlag : "");
//        inboundBuilder.putMetadata(Osx.Metadata.JobId.name(), jobId != null ? jobId : "");
//        inboundBuilder.putMetadata(Osx.Metadata.MessageCode.name(), messageCode != null ? messageCode : "");
//        inboundBuilder.putMetadata(Osx.Metadata.RetryCount.name(), retryCount != null ? retryCount : "");
        return inboundBuilder;
    }


    static public Map parseHttpHeader(OsxContext  context ) {

//        static public  final String   Version="x-ptp-version";
//        static public  final String   TechProviderCode  = "x-ptp-tech-provider-code";
//        static public  final String   TraceID = "x-ptp-trace-id";
//        static public  final String   Token = "x-ptp-token";
//        static public  final String   Uri= "x-ptp-uri";
//        static public  final String   FromNodeID = "x-ptp-from-node-id";
//        static public  final String   FromInstID = "x-ptp-from-inst-id";
//        static public  final String   TargetNodeID = "x-ptp-target-node-id";
//        static public  final String   TargetInstID = "x-ptp-target-inst-id";
//        static public  final String   SessionID = "x-ptp-session-id";
//        static public  final String   MessageTopic = "x-ptp-topic";


        Map header = Maps.newHashMap();
        header.put(PtpHttpHeader.Version,context.getVersion());
        header.put(PtpHttpHeader.TechProviderCode,context.getTechProviderCode());
        header.put(PtpHttpHeader.TraceID,context.getTraceId());
        header.put(PtpHttpHeader.Token,context.getToken());
        header.put(PtpHttpHeader.Uri,context.getUri());
        header.put(PtpHttpHeader.FromNodeID,context.getSrcNodeId());
        header.put(PtpHttpHeader.FromInstID,context.getSrcInstId());
        header.put(PtpHttpHeader.TargetNodeID,context.getDesNodeId());
        header.put(PtpHttpHeader.TargetInstID,context.getDesInstId());
        header.put(PtpHttpHeader.SessionID,context.getSessionId());
        header.put(PtpHttpHeader.MessageTopic,context.getTopic());
        header.put(PtpHttpHeader.QueueType,context.getQueueType());
        header.put(PtpHttpHeader.MessageFlag,context.getMessageFlag());

//        header.put(PtpHttpHeader.Version, version != null ? version : "");
//        header.put(PtpHttpHeader.TechProviderCode, techProviderCode != null ? techProviderCode : "");
//        header.put(PtpHttpHeader.TraceID, traceId != null ? traceId : "");
//        header.put(PtpHttpHeader.Token, token != null ? token : "");
//        header.put(PtpHttpHeader.SourceNodeID, sourceNodeId != null ? sourceNodeId : "");
//        header.put(PtpHttpHeader.TargetNodeID, targetNodeId != null ? targetNodeId : "");
//        header.put(PtpHttpHeader.SourceInstID, sourceInstId != null ? sourceInstId : "");
//        header.put(PtpHttpHeader.TargetInstID, targetInstId != null ? targetInstId : "");
//        header.put(PtpHttpHeader.SessionID, sessionId != null ? sessionId : "");
//        header.put(PtpHttpHeader.MessageTopic, topic != null ? topic : "");
//        header.put(PtpHttpHeader.MessageCode, messageCode);
//        header.put(PtpHttpHeader.SourceComponentName, sourceComponentName != null ? sourceComponentName : "");
//        header.put(PtpHttpHeader.TargetComponentName, targetComponentName != null ? targetComponentName : "");
//        header.put(PtpHttpHeader.TargetMethod, targetMethod != null ? targetMethod : "");
//        header.put(PtpHttpHeader.SourceMethod, sourceMethod != null ? sourceMethod : "");
//        header.put(PtpHttpHeader.MessageOffSet, offsetString != null ? offsetString : "");
//        header.put(PtpHttpHeader.InstanceId, InstanceId != null ? InstanceId : "");
//        header.put(PtpHttpHeader.Timestamp, timestamp != null ? timestamp : "");
//        header.put(PtpHttpHeader.MessageFlag, messageFlag != null ? messageFlag : "");
//        header.put(PtpHttpHeader.JobId, jobId != null ? jobId : "");

        return header;
    }

    /**
     * z
     * @param context
     * @param produceRequest
     * @param routerInfo
     * @param usePooled
     * @return
     */
    static public Osx.TransportOutbound redirectPush(OsxContext context, Osx.PushInbound
            produceRequest, RouterInfo routerInfo, boolean usePooled) {
            Osx.TransportOutbound result = null;
            PrivateTransferTransportGrpc.PrivateTransferTransportBlockingStub stub = null;
            if (context.getData(Dict.BLOCKING_STUB) == null) {
                ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo, usePooled);
                stub = PrivateTransferTransportGrpc.newBlockingStub(managedChannel);
            } else {
                stub = (PrivateTransferTransportGrpc.PrivateTransferTransportBlockingStub) context.getData(Dict.BLOCKING_STUB);
            }
            try {
                result = stub.push(produceRequest);
            } catch (StatusRuntimeException e) {
                logger.error("redirect error", e);
                throw new RemoteRpcException(StatusCode.NET_ERROR, "send to " + routerInfo.toKey() + " error : " + e.getMessage());
            }
            return result;
    }




    static public Osx.TransportOutbound redirectHttpPush(OsxContext context, Osx.PushInbound
            produceRequest, RouterInfo routerInfo) {
        Osx.TransportOutbound  result =null;
        String url = routerInfo.getUrl();
        Map header = parseHttpHeader(context);
        long startTime = System.currentTimeMillis();
        HttpDataWrapper  httpDataWrapper=null;
        try {
            if (routerInfo.getProtocol().equals(Protocol.http)) {

                if (routerInfo.isUseSSL()) {
                    httpDataWrapper = HttpsClientPool.sendPostWithCert(url, produceRequest.getPayload().toByteArray(), header, routerInfo.getCaFile(), routerInfo.getCertChainFile(), routerInfo.getPrivateKeyFile());
                } else {
                    httpDataWrapper = HttpClientPool.sendPost(url, produceRequest.getPayload().toByteArray(), header);
                }
            }
        } catch (Exception e) {
//            e.printStackTrace();
            logger.error("sendPtpPost failed : url = {}, startTime = {}  , cost = {} ,header = {} , body = {} \n"
                    , url, startTime, System.currentTimeMillis() - startTime, JsonUtil.object2Json(header), JsonUtil.object2Json(produceRequest.getPayload()), e);
            ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
            result = Osx.TransportOutbound.newBuilder().setCode(exceptionInfo.getCode()).setMessage(exceptionInfo.getMessage()).build();
        }
        if(httpDataWrapper!=null) {
            try {
                result = Osx.TransportOutbound.parseFrom(ByteString.copyFrom(httpDataWrapper.getPayload()));
            } catch (InvalidProtocolBufferException e) {
                e.printStackTrace();
            }
        }
        return  result;
    }

    static public Osx.TransportOutbound redirectPop(OsxContext context, RouterInfo routerInfo, Osx.PopInbound inbound) {
        ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo,true);
        context.setActionType(ActionType.REDIRECT_CONSUME.name());
        PrivateTransferTransportGrpc.PrivateTransferTransportBlockingStub stub = PrivateTransferTransportGrpc.newBlockingStub(managedChannel);
        return stub.pop(inbound);
    }

    static public Osx.TransportOutbound redirectPeek(OsxContext context, RouterInfo routerInfo, Osx.PeekInbound inbound) {
        ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo,true);
        context.setActionType(ActionType.REDIRECT_CONSUME.name());
        PrivateTransferTransportGrpc.PrivateTransferTransportBlockingStub stub = PrivateTransferTransportGrpc.newBlockingStub(managedChannel);
        return stub.peek(inbound);
    }

    static public Osx.TransportOutbound  redirectRelease(OsxContext  context,RouterInfo routerInfo,Osx.ReleaseInbound  inbound){
        ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo,true);
        context.setActionType(ActionType.CANCEL_TOPIC.name());
        PrivateTransferTransportGrpc.PrivateTransferTransportBlockingStub stub = PrivateTransferTransportGrpc.newBlockingStub(managedChannel);
        return stub.release(inbound);
    }

    static public Object redirect(OsxContext context, Object
            data, RouterInfo routerInfo, boolean usePooled) {
        AssertUtil.notNull(routerInfo, context.getDesNodeId() != null ? "des partyId " + context.getDesNodeId() + " router info is null" : " error router info");
        Osx.Outbound result = null;

        if (routerInfo.isCycle()) {
            throw new CycleRouteInfoException("cycle router info");
        }
        if (routerInfo.getProtocol() == null ||context.getProtocol().equals(Protocol.grpc)&& routerInfo.getProtocol().equals(Protocol.grpc)) {
            //来自旧版fateflow的请求，需要用旧版的stub
            Osx.Inbound inbound = (Osx.Inbound)data;
            context.setDataSize(inbound.getSerializedSize());

//            if (context.isDestination() && Role.fateflow.name().equals(routerInfo.getDesRole())
//                    && SourceMethod.OLDUNARY_CALL.name().equals(produceRequest.getMetadataMap().get(Osx.Metadata.SourceMethod.name()))) {
//                ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo, usePooled);
//                DataTransferServiceGrpc.DataTransferServiceBlockingStub stub = DataTransferServiceGrpc.newBlockingStub(managedChannel);
//                Proxy.Packet request;
//                try {
//                    request = Proxy.Packet.parseFrom(produceRequest.getPayload().toByteArray());
//                } catch (InvalidProtocolBufferException e) {
//                    throw new RuntimeException(e);
//                }
//                Proxy.Packet response = stub.unaryCall(request);
//                result = Osx.Outbound.newBuilder().setPayload(response.toByteString()).setCode(StatusCode.SUCCESS).build();
//            } else
            {

                PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub stub = null;
                if (context.getData(Dict.BLOCKING_STUB) == null) {
                    ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo, usePooled);
                    stub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
                } else {
                    stub = (PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub) context.getData(Dict.BLOCKING_STUB);
                }
                try {
                    result = stub.invoke(inbound);
                } catch (StatusRuntimeException e) {
                    logger.error("redirect error", e);
                    throw new RemoteRpcException(StatusCode.NET_ERROR, "send to " + routerInfo.toKey() + " error : " + e.getMessage());
                }
            }
            // ServiceContainer.tokenApplyService.applyToken(context,routerInfo.getResource(),produceRequest.getSerializedSize());
        } else {

            HttpInvoke  httpInvoke = (HttpInvoke)data;
            String url = routerInfo.getUrl();
            Map header = parseHttpHeader(context);

            long startTime = System.currentTimeMillis();
            try {
                if (routerInfo.getProtocol().equals(Protocol.http)) {
                    HttpDataWrapper httpDataWrapper =null;
                    if (routerInfo.isUseSSL()) {
                        httpDataWrapper= HttpsClientPool.sendPostWithCert(url, JsonUtil.object2Json(httpInvoke).getBytes(StandardCharsets.UTF_8), header, routerInfo.getCaFile(), routerInfo.getCertChainFile(), routerInfo.getPrivateKeyFile());
                    } else {
                         httpDataWrapper = HttpClientPool.sendPost(url,  JsonUtil.object2Json(httpInvoke).getBytes(StandardCharsets.UTF_8), header);
                    }
                    if(httpDataWrapper!=null) {
                        HttpInvokeResult  httpInvokeResult   = JsonUtil.json2Object(httpDataWrapper.getPayload(), HttpInvokeResult.class );
                        return  httpInvokeResult;
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
//                logger.error("sendPtpPost failed : url = {}, startTime = {}  , cost = {} ,header = {} , body = {} \n"
//                        , url, startTime, System.currentTimeMillis() - startTime, JsonUtil.object2Json(header), , e);
//                ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
//                result = Osx.Outbound.newBuilder().setCode(exceptionInfo.getCode()).setMessage(exceptionInfo.getMessage()).build();
                throw  new RemoteRpcException(e);
            }
        }
        return result;
    }


    public static Osx.Outbound.Builder buildResponseInner(String code, String msgReturn, byte[] content) {

        Osx.Outbound.Builder builder = Osx.Outbound.newBuilder();
        builder.setCode(code);
        builder.setMessage(msgReturn);
        if (content != null) {
            builder.setPayload(ByteString.copyFrom(content));
        }
        return builder;
    }

    public static Osx.TransportOutbound buildTransportOutbound(String code, String msgReturn, TransferQueueConsumeResult messageWraper) {
        byte[] content = null;
        if(messageWraper!=null&&messageWraper.getMessage()!=null){
            content =messageWraper.getMessage().getBody();
        }
        Osx.TransportOutbound.Builder builder = Osx.TransportOutbound.newBuilder();
        builder.setCode(code);
        builder.setMessage(msgReturn);
        if (content != null) {
            builder.setPayload(ByteString.copyFrom(content));
        }
        return builder.build();
    }


//    public static Osx.Outbound buildTransResponse(String code, String msgReturn, TransferQueue.TransferQueueConsumeResult messageWraper) {
//
//        byte[] content = null;
//        if (messageWraper != null) {
//            Osx.Message message = null;
//            try {
//                message = Osx.Message.parseFrom(messageWraper.getMessage().getBody());
//            } catch (InvalidProtocolBufferException e) {
//                logger.error("parse message error", e);
//            }
//            content = message.toByteArray();
//        }
//        Osx.Outbound.Builder builder = buildResponseInner(code, msgReturn, content);
//        if (messageWraper != null) {
//            builder.putMetadata(Osx.Metadata.MessageOffSet.name(), Long.toString(messageWraper.getRequestIndex()));
//        }
//        return builder.build();
//    }



    public static Osx.Outbound buildResponse(String code, String msgReturn, TransferQueueConsumeResult messageWraper) {

        byte[] content = null;
        if (messageWraper != null) {
            Osx.Message message = null;
            try {
                message = Osx.Message.parseFrom(messageWraper.getMessage().getBody());
            } catch (InvalidProtocolBufferException e) {
                logger.error("parse message error", e);
            }
            content = message.toByteArray();
        }
        Osx.Outbound.Builder builder = buildResponseInner(code, msgReturn, content);
        if (messageWraper != null) {
            builder.putMetadata(Osx.Metadata.MessageOffSet.name(), Long.toString(messageWraper.getRequestIndex()));
        }
        return builder.build();
    }

    public static void checkResponse(Osx.Outbound outbound) {
        if (outbound != null) {
            String code = outbound.getCode();
            String message = outbound.getMessage();
            if (!StatusCode.PTP_SUCCESS.equals(code)) {
                logger.error("================== xxxxxx  {}", outbound);
                throw new RemoteRpcException("remote code : " + code + " remote msg: " + message);
            }
        } else {
            throw new RemoteRpcException("has no response");
        }
    }

    public static void writeHttpRespose(ServletResponse response, String code,
                                        String msg,
                                        byte[] content) {
        try {
//            response.setHeader(PtpHttpHeader.ReturnCode, code);
//            response.setHeader(PtpHttpHeader.MessageCode, msg);
            OutputStream outputStream = response.getOutputStream();
            if (content != null) {
                System.err.println("return data :"+new String(content));
                outputStream.write(content);
            }
            outputStream.flush();
        } catch (IOException e) {
            logger.error("write http response error", e);
        }
    }







    public static void main(String[] args) {

//       System.err.println( TransferUtil.parseUri("/testuri"));
//       System.err.println(TransferUtil.buildUrl("grpcs://","yyyy.com","/uuuuu"));
//        MBeanServer platformMBeanServer = ManagementFactory.getPlatformMBeanServer();
//
//        if (platformMBeanServer instanceof com.sun.management.OperatingSystemMXBean) {
//            com.sun.management.OperatingSystemMXBean osBean = (com.sun.management.OperatingSystemMXBean) platformMBeanServer;
//
//            // 获取连接数
//            int connectionCount = osBean.getAvailableProcessors();
//            System.out.println("HTTP 连接数: " + connectionCount);
//        } else {
//            System.out.println("当前平台不支持获取 HTTP 连接数");
//        }

//        TransferUtil a = new TransferUtil();
//        a.testHttps();
    }

//    public void testHttps() {
//        try {
//            new Thread(() -> {
//                Osx.Outbound outbound = null;
//                try {
//                    Thread.sleep(3000);
//                    outbound = HttpsClientPool.sendPtpPost("https://127.0.0.1:8088/osx/inbound", new byte[10], null, "D:\\22\\ca.crt", "D:\\22\\174_2.crt", "D:\\22\\174_2.key");
//                } catch (Exception e) {
//                    e.printStackTrace();
//                }
//                System.out.println("outbound = " + outbound);
//
//            }).start();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }

    public  static  byte[] read(InputStream input) throws IOException {

        byte[] result = null;
        byte[] split = new byte[1024];
        int length=0;
        int count;
        while( (count = input.read(split))!=-1){
            byte[] temp =new  byte[length+count];
            System.arraycopy(split, 0, temp, length, count);
            if(result!=null) {
                System.arraycopy(result, 0,temp,0,length );
            }
            result =  temp;
            length =  result.length;
        }
        return result;
    }
}
