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
package com.osx.broker.util;


import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.osx.broker.eggroll.ErRollSiteHeader;
import com.osx.broker.http.HttpClientPool;
import com.osx.broker.http.PtpHttpResponse;
import com.osx.broker.queue.TransferQueue;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.Dict;
import com.osx.core.constant.Protocol;
import com.osx.core.constant.PtpHttpHeader;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.Context;
import com.osx.core.exceptions.ConfigErrorException;
import com.osx.core.exceptions.NoRouterInfoException;
import com.osx.core.exceptions.RemoteRpcException;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.router.RouterInfo;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import org.apache.commons.lang3.StringUtils;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;

import javax.servlet.http.HttpServletRequest;
import java.util.Map;

public class TransferUtil {

    /**
     * 2.0之前版本
     *
     * @param version
     * @return
     */
    public static boolean isOldVersionFate(String version) {

        try{
            if (StringUtils.isEmpty(version))
                version= MetaInfo.PROPERTY_DEFAULT_CLIENT_VERSION;
            String firstVersion = version.substring(0,1);
            if (Integer.parseInt(firstVersion) >= 2) {
                return false;
            } else {
                return true;
            }
        }catch(NumberFormatException e){
            throw new ConfigErrorException("remote version config error : "+version);
        }

    }


    public static String buildResource(Osx.Inbound inbound){
        String  sourceNodeId = inbound.getMetadataMap().get(Osx.Header.SourceNodeID.name());
        String  targetNodeId = inbound.getMetadataMap().get(Osx.Header.TargetNodeID.name());
        String  sourceInstId = inbound.getMetadataMap().get(Osx.Header.SourceInstID.name());
        if(sourceInstId==null){
            sourceInstId="";
        }
        String  targetInstId = inbound.getMetadataMap().get(Osx.Header.TargetInstID.name());
        if(targetInstId==null){
            targetInstId="";
        }
        StringBuffer  sb =  new StringBuffer();
        sb.append(sourceInstId).append(sourceNodeId).append("_").append(targetInstId).append(targetNodeId);
        return  sb.toString();
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

    public static Proxy.Packet parsePacketFromInbound(Osx.Inbound inbound){
        try {
           return  Proxy.Packet.parseFrom(inbound.getPayload());
        } catch (InvalidProtocolBufferException e) {
            return null;
        }
    }

    public static Osx.Inbound buildInboundFromPushingPacket(Proxy.Packet packet, String targetMethod) {
        Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
        Proxy.Topic srcTopic = packet.getHeader().getSrc();
        String srcPartyId = srcTopic.getPartyId();
        Proxy.Metadata metadata = packet.getHeader();
        ByteString encodedRollSiteHeader = metadata.getExt();
        ErRollSiteHeader rsHeader = null;
        try {
            rsHeader = ErRollSiteHeader.parseFromPb(Transfer.RollSiteHeader.parseFrom(encodedRollSiteHeader));
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }

        String sessionId = "";
        if (rsHeader != null) {
            sessionId = String.join("_", rsHeader.getRollSiteSessionId(), rsHeader.getDstRole(), rsHeader.getDstPartyId());
        }
        Proxy.Topic desTopic = packet.getHeader().getDst();
        String desPartyId = desTopic.getPartyId();
        String desRole = desTopic.getRole();
        inboundBuilder.setPayload(packet.toByteString());
        inboundBuilder.putMetadata(Osx.Header.Version.name(), Long.toString(MetaInfo.CURRENT_VERSION));
        inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(),  MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        inboundBuilder.putMetadata(Osx.Header.Token.name(), "");
        inboundBuilder.putMetadata(Osx.Header.SourceNodeID.name(), srcPartyId);
        inboundBuilder.putMetadata(Osx.Header.TargetNodeID.name(), desPartyId);
        inboundBuilder.putMetadata(Osx.Header.SourceInstID.name(), "");
        inboundBuilder.putMetadata(Osx.Header.TargetInstID.name(), "");
        inboundBuilder.putMetadata(Osx.Header.SessionID.name(), sessionId);
        inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), targetMethod);
        inboundBuilder.putMetadata(Osx.Metadata.TargetComponentName.name(), desRole);
        inboundBuilder.putMetadata(Osx.Metadata.SourceComponentName.name(), "");
        return inboundBuilder.build();
    };

    static  public void buildHttpFromPb(Osx.Inbound  inbound){




    }


    static  public Osx.Inbound.Builder  buildPbFromHttpRequest(HttpServletRequest request){

        Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
        String Version = request.getHeader(PtpHttpHeader.Version);
        String TechProviderCode = request.getHeader(PtpHttpHeader.TechProviderCode);
        String TraceID = request.getHeader(PtpHttpHeader.TraceID);
        String Token = request.getHeader(PtpHttpHeader.Token);
        String SourceNodeID = request.getHeader(PtpHttpHeader.SourceNodeID);
        String TargetNodeID = request.getHeader(PtpHttpHeader.TargetNodeID);
        String SourceInstID = request.getHeader(PtpHttpHeader.SourceInstID);
        String TargetInstID = request.getHeader(PtpHttpHeader.TargetInstID);
        String SessionID = request.getHeader(PtpHttpHeader.SessionID);
        String MessageTopic = request.getHeader(PtpHttpHeader.MessageTopic);
        String MessageCode = request.getHeader(PtpHttpHeader.MessageCode);
        String SourceComponentName = request.getHeader(PtpHttpHeader.SourceComponentName);
        String TargetComponentName = request.getHeader(PtpHttpHeader.TargetComponentName);
        String TargetMethod = request.getHeader(PtpHttpHeader.TargetMethod);
        String MessageOffSet = request.getHeader(PtpHttpHeader.MessageOffSet);
        String InstanceId = request.getHeader(PtpHttpHeader.InstanceId);
        String Timestamp = request.getHeader(PtpHttpHeader.Timestamp);

        inboundBuilder.putMetadata(Osx.Header.Version.name(), Version != null ? Version : "");
        inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(), TechProviderCode != null ? TechProviderCode : "");
        inboundBuilder.putMetadata(Osx.Header.Token.name(), Token != null ? Token : "");
        inboundBuilder.putMetadata(Osx.Header.SourceNodeID.name(), SourceNodeID != null ? SourceNodeID : "");
        inboundBuilder.putMetadata(Osx.Header.TargetNodeID.name(), TargetNodeID != null ? TargetNodeID : "");
        inboundBuilder.putMetadata(Osx.Header.SourceInstID.name(), SourceInstID != null ? SourceInstID : "");
        inboundBuilder.putMetadata(Osx.Header.TargetInstID.name(), TargetInstID != null ? TargetInstID : "");
        inboundBuilder.putMetadata(Osx.Header.SessionID.name(), SessionID != null ? SessionID : "");
        inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), TargetMethod != null ? TargetMethod : "");
        inboundBuilder.putMetadata(Osx.Metadata.TargetComponentName.name(), TargetComponentName != null ? TargetComponentName : "");
        inboundBuilder.putMetadata(Osx.Metadata.SourceComponentName.name(), SourceComponentName != null ? SourceComponentName : "");
        inboundBuilder.putMetadata(Osx.Metadata.MessageTopic.name(), MessageTopic != null ? MessageTopic : "");
        inboundBuilder.putMetadata(Osx.Metadata.MessageOffSet.name(), MessageOffSet != null ? MessageOffSet : "");
        inboundBuilder.putMetadata(Osx.Metadata.InstanceId.name(), InstanceId != null ? InstanceId : "");
        inboundBuilder.putMetadata(Osx.Metadata.Timestamp.name(), Timestamp != null ? Timestamp : "");
        return  inboundBuilder;


    }



    static public Osx.Outbound redirect(Context context, Osx.Inbound
            produceRequest, RouterInfo routerInfo, boolean forceSend) {
        Osx.Outbound result = null;
        // context.setActionType("redirect");
        // 目的端协议为grpc
        if (routerInfo == null) {
            throw new NoRouterInfoException("can not find router info");
        }
        if (routerInfo.getProtocol() == null || routerInfo.getProtocol().equals(Protocol.GRPC)) {
            ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo,true);
            PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub stub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
            try {
                result = stub.invoke(produceRequest);
            } catch (StatusRuntimeException e) {
                throw new RemoteRpcException(StatusCode.NET_ERROR, "send to " + routerInfo.toKey() + " error");
            }
            // ServiceContainer.tokenApplyService.applyToken(context,routerInfo.getResource(),produceRequest.getSerializedSize());
        }else{
            if(routerInfo.getProtocol().equals(Protocol.HTTP)){
                String url = routerInfo.getUrl();

                Map<String, String> metaDataMap = produceRequest.getMetadataMap();

                String version = metaDataMap.get(Osx.Header.Version.name());
                String techProviderCode = metaDataMap.get(Osx.Header.TechProviderCode.name());
                String traceId = metaDataMap.get(Osx.Header.TraceID.name());
                String token = metaDataMap.get(Osx.Header.Token.name());
                String sourceNodeId = metaDataMap.get(Osx.Header.SourceNodeID.name());
                String targetNodeId = metaDataMap.get(Osx.Header.TargetNodeID.name());
                String sourceInstId = metaDataMap.get(Osx.Header.SourceInstID.name());
                String targetInstId = metaDataMap.get(Osx.Header.TargetInstID.name());
                String sessionId = metaDataMap.get(Osx.Header.SessionID.name());
                String targetMethod = metaDataMap.get(Osx.Metadata.TargetMethod.name());
                String targetComponentName = metaDataMap.get(Osx.Metadata.TargetComponentName.name());
                String sourceComponentName = metaDataMap.get(Osx.Metadata.SourceComponentName.name());
                String sourcePartyId = StringUtils.isEmpty(sourceInstId) ? sourceNodeId : sourceInstId + "." + sourceNodeId;
                String targetPartyId = StringUtils.isEmpty(targetInstId) ? targetNodeId : targetInstId + "." + targetNodeId;
                String topic = metaDataMap.get(Osx.Metadata.MessageTopic.name());
                String offsetString = metaDataMap.get(Osx.Metadata.MessageOffSet.name());
                String InstanceId = metaDataMap.get(Osx.Metadata.InstanceId.name());
                String timestamp = metaDataMap.get(Osx.Metadata.Timestamp.name());
                String messageCode = metaDataMap.get(Osx.Metadata.MessageCode.name());
                Map header = Maps.newHashMap();
                header.put(PtpHttpHeader.Version,version!=null?version:"");
                header.put(PtpHttpHeader.TechProviderCode,techProviderCode!=null?techProviderCode:"");
                header.put(PtpHttpHeader.TraceID,traceId!=null?traceId:"");
                header.put(PtpHttpHeader.Token,token!=null?token:"");
                header.put(PtpHttpHeader.SourceNodeID,sourceNodeId!=null?sourceNodeId:"");
                header.put(PtpHttpHeader.TargetNodeID,targetNodeId!=null?targetNodeId:"");
                header.put(PtpHttpHeader.SourceInstID,sourceInstId!=null?sourceInstId:"");
                header.put(PtpHttpHeader.TargetInstID,targetInstId!=null?targetInstId:"");
                header.put(PtpHttpHeader.SessionID,sessionId!=null?sessionId:"");
                header.put(PtpHttpHeader.MessageTopic,topic!=null?topic:"");
                header.put(PtpHttpHeader.MessageCode,messageCode);
                header.put(PtpHttpHeader.SourceComponentName,sourceComponentName!=null?sourceComponentName:"");
                header.put(PtpHttpHeader.TargetComponentName,targetComponentName!=null?targetComponentName:"");
                header.put(PtpHttpHeader.TargetMethod,targetMethod!=null?targetMethod:"");
                header.put(PtpHttpHeader.MessageOffSet,offsetString!=null?offsetString:"");
                header.put(PtpHttpHeader.InstanceId,InstanceId!=null?InstanceId:"");
                header.put(PtpHttpHeader.Timestamp,timestamp!=null?timestamp:"");
                result = HttpClientPool.sendPtpPost(url,produceRequest.getPayload().toByteArray(),header);
            }
        }

        return result;

    }


    public static Osx.Outbound buildResponse(String code, String msgReturn, TransferQueue.TransferQueueConsumeResult messageWraper) {
        // FireworkTransfer.ConsumeResponse.Builder  consumeResponseBuilder = FireworkTransfer.ConsumeResponse.newBuilder();
        Osx.Outbound.Builder builder = Osx.Outbound.newBuilder();

        builder.setCode(code);
        builder.setMessage(msgReturn);
        if (messageWraper != null) {
            Osx.Message message = null;
            try {
                message = Osx.Message.parseFrom(messageWraper.getMessage().getBody());
            } catch (InvalidProtocolBufferException e) {
                e.printStackTrace();
            }
            builder.setPayload(message.toByteString());
            builder.putMetadata(Osx.Metadata.MessageOffSet.name(), Long.toString(messageWraper.getRequestIndex()));
//                FireworkTransfer.Message msg = produceRequest.getMessage();
//                consumeResponseBuilder.setTransferId(produceRequest.getTransferId());
//                consumeResponseBuilder.setMessage(msg);
//                consumeResponseBuilder.setStartOffset(messageWraper.getRequestIndex());
//                consumeResponseBuilder.setTotalOffset(messageWraper.getLogicIndexTotal());
        }

        return builder.build();
    }


    public static  void main(String[] args){
        System.err.println(isOldVersionFate(null));
    }
}
