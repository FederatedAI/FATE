package com.osx.broker.util;

//import com.firework.cluster.rpc.FireworkTransfer;
import com.google.protobuf.ByteString;
import com.osx.core.constant.StatusCode;
import com.osx.core.exceptions.RemoteRpcException;
import com.osx.core.frame.*;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.Protocol;
import com.osx.core.context.Context;
import com.osx.core.exceptions.NoRouterInfoException;
import com.osx.core.router.RouterInfo;
import com.osx.federation.rpc.Osx;
import com.osx.broker.eggroll.ErRollSiteHeader;
import com.osx.broker.queue.TransferQueue;
import com.google.protobuf.InvalidProtocolBufferException;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import org.ppc.ptp.Pcp;
import org.ppc.ptp.PrivateTransferProtocolGrpc;

public class TransferUtil {

    /**
     * 2.0之前版本
     * @param version
     * @return
     */
    public static boolean isOldVersionFate(String version){
        if(version==null)
            return true;
        int versionInteger = Integer.parseInt(version);
        if(versionInteger>=200){
            System.err.println("isOldVersionFate return false");
            return false;
        }else{
            System.err.println("isOldVersionFate return true");
            return true;
        }

    }


    public  static Proxy.Metadata  buildProxyMetadataFromOutbound(Pcp.Outbound  outbound){
        try {
          return   Proxy.Metadata.parseFrom(outbound.getPayload());
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }
        return null;
    };

    public  static Pcp.Inbound  buildInboundFromPushingPacket(Proxy.Packet  packet,String  targetMethod){
        System.err.println("==================buildInboundFromPushingPacket==========================");
        Pcp.Inbound.Builder  inboundBuilder = Pcp.Inbound.newBuilder();
        Proxy.Topic srcTopic =packet.getHeader().getSrc();
        String  srcPartyId = srcTopic.getPartyId();
        Proxy.Metadata metadata = packet.getHeader();
       // String oneLineStringMetadata = ToStringUtils.toOneLineString(metadata);
        ByteString encodedRollSiteHeader = metadata.getExt();
        //context.setActionType("push-eggroll");
        ErRollSiteHeader rsHeader=null;
        try {
            rsHeader= ErRollSiteHeader.parseFromPb(Transfer.RollSiteHeader.parseFrom(encodedRollSiteHeader));


        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();

        }
        //logger.info("=========ErRollSiteHeader {}",rsHeader);
        //"#", prefix: Array[String] = Array("__rsk")
        String sessionId ="";
        if(rsHeader!=null) {
              sessionId = String.join("_", rsHeader.getRollSiteSessionId() , rsHeader.getDstRole(), rsHeader.getDstPartyId());
        }
        Proxy.Topic desTopic = packet.getHeader().getDst();
        String desPartyId = desTopic.getPartyId();
        String desRole = desTopic.getRole();
        inboundBuilder.setPayload(packet.toByteString());
        inboundBuilder.putMetadata(Pcp.Header.Version.name(), Long.toString(MetaInfo.CURRENT_VERSION));
        inboundBuilder.putMetadata(Pcp.Header.TechProviderCode.name(),"FT");
        inboundBuilder.putMetadata(Pcp.Header.Token.name(),"testToken");
        inboundBuilder.putMetadata(Pcp.Header.SourceNodeID.name(),srcPartyId);
        inboundBuilder.putMetadata(Pcp.Header.TargetNodeID.name(),desPartyId);
        inboundBuilder.putMetadata(Pcp.Header.SourceInstID.name(),"");
        inboundBuilder.putMetadata(Pcp.Header.TargetInstID.name(),"");
        inboundBuilder.putMetadata(Pcp.Header.SessionID.name(),sessionId);
        inboundBuilder.putMetadata(Pcp.Metadata.TargetMethod.name(), targetMethod);
        inboundBuilder.putMetadata(Pcp.Metadata.TargetComponentName.name(),desRole);
        inboundBuilder.putMetadata(Pcp.Metadata.SourceComponentName.name(),"");
        return  inboundBuilder.build();
        //inboundBuilder.putMetadata(Pcp.Metadata.MessageTopic.name(),transferId);



    };



    static public  Pcp.Outbound  redirect(Context context , Pcp.Inbound
            produceRequest, RouterInfo routerInfo, boolean forceSend){
        Pcp.Outbound result = null;
       // context.setActionType("redirect");
        // 目的端协议为grpc
        if(routerInfo==null){
            throw  new NoRouterInfoException("");
        }
        if(routerInfo.getProtocol()==null||routerInfo.getProtocol().equals(Protocol.GRPC)) {
            ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo);
            PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub stub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);

            try {
                result = stub.invoke(produceRequest);
            }catch (StatusRuntimeException e){
                throw  new RemoteRpcException(StatusCode.NET_ERROR,"send to "+routerInfo.toKey()+" error");
            }
            // ServiceContainer.tokenApplyService.applyToken(context,routerInfo.getResource(),produceRequest.getSerializedSize());
        }

        return  result;

    }


    public static Pcp.Outbound buildResponse(String code , String msgReturn , TransferQueue.TransferQueueConsumeResult messageWraper){
       // FireworkTransfer.ConsumeResponse.Builder  consumeResponseBuilder = FireworkTransfer.ConsumeResponse.newBuilder();
        Pcp.Outbound.Builder  builder = Pcp.Outbound.newBuilder();

            builder.setCode(code);
            builder.setMessage(msgReturn);
            if(messageWraper!=null) {
                Osx.Message message = null;
                try {
                    message = Osx.Message.parseFrom(messageWraper.getMessage().getBody());
                } catch (InvalidProtocolBufferException e) {
                    e.printStackTrace();
                }
                builder.setPayload(message.toByteString());
                builder.putMetadata(Pcp.Metadata.MessageOffSet.name(),Long.toString(messageWraper.getRequestIndex()) );
//                FireworkTransfer.Message msg = produceRequest.getMessage();
//                consumeResponseBuilder.setTransferId(produceRequest.getTransferId());
//                consumeResponseBuilder.setMessage(msg);
//                consumeResponseBuilder.setStartOffset(messageWraper.getRequestIndex());
//                consumeResponseBuilder.setTotalOffset(messageWraper.getLogicIndexTotal());
            }

        return builder.build();
    }
}
