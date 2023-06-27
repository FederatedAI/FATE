package com.osx.broker.test.grpc;
//import com.firework.cluster.rpc.FireworkQueueServiceGrpc;
//import com.firework.cluster.rpc.FireworkTransfer;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.Dict;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.ptp.TargetMethod;

import com.osx.core.utils.JsonUtil;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import org.junit.Before;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class QueueTest {
    Logger logger = LoggerFactory.getLogger(QueueTest.class);
    String ip = "localhost";
    //int port = 8250;//nginx
    int port = 9370;//nginx
    String desPartyId = "9999";
    String desRole = "";
    String srcPartyId = "10000";
    String srcRole = "";
    String transferId = "testTransferId";
    String sessionId = "testSessionId";
    PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub blockingStub;
//    FireworkQueueServiceGrpc.FireworkQueueServiceBlockingStub  blockingStub;

    public static ManagedChannel createManagedChannel(String ip, int port) {
        try {
            NettyChannelBuilder channelBuilder = NettyChannelBuilder
                    .forAddress(ip, port)
                    .keepAliveTime(12, TimeUnit.MINUTES)
                    .keepAliveTimeout(12, TimeUnit.MINUTES)
                    .keepAliveWithoutCalls(true)
                    //.idleTimeout(60, TimeUnit.SECONDS)
                    .perRpcBufferLimit(128 << 20)
                    .flowControlWindow(32 << 20)
                    .maxInboundMessageSize(32 << 20)
                    .enableRetry()
                    .retryBufferSize(16 << 20)
                    .maxRetryAttempts(20);

            channelBuilder.usePlaintext();
            return channelBuilder.build();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    @Before
    public void init() {
        ManagedChannel managedChannel = createManagedChannel(ip, port);
        //  stub =      PrivateTransferProtocolGrpc.newBlockingStub();
        ManagedChannel managedChannel2 = createManagedChannel(ip, port);
        blockingStub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel2);
    }




    @Test
    public void test02Query() {
        Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
        inboundBuilder.putMetadata(Osx.Header.Version.name(), "123");
        inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(),  MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        inboundBuilder.putMetadata(Osx.Header.Token.name(), "testToken");
        inboundBuilder.putMetadata(Osx.Header.SourceNodeID.name(), "9999");
        inboundBuilder.putMetadata(Osx.Header.TargetNodeID.name(), "10000");
        inboundBuilder.putMetadata(Osx.Header.SourceInstID.name(), "");
        inboundBuilder.putMetadata(Osx.Header.TargetInstID.name(), "");
        inboundBuilder.putMetadata(Osx.Header.SessionID.name(), "testSessionID");
        inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), TargetMethod.QUERY_TOPIC.name());
        inboundBuilder.putMetadata(Osx.Metadata.TargetComponentName.name(), "");
        inboundBuilder.putMetadata(Osx.Metadata.SourceComponentName.name(), "");
        inboundBuilder.putMetadata(Osx.Metadata.MessageTopic.name(), transferId);


        Osx.Outbound outbound = blockingStub.invoke(inboundBuilder.build());
        Osx.TopicInfo topicInfo = null;
        try {
            topicInfo = Osx.TopicInfo.parseFrom(outbound.getPayload());
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }
        System.err.println("response " + topicInfo);
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }


    public void testUnaryConsume(){

    }


    @Test
    public void test04UnaryProduce() {
        for (int i = 0; i < 1; i++) {
            Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
            inboundBuilder.putMetadata(Osx.Header.Version.name(), "123");
            inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(),  MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
            inboundBuilder.putMetadata(Osx.Header.Token.name(), "testToken");
            inboundBuilder.putMetadata(Osx.Header.SourceNodeID.name(), "9999");
            inboundBuilder.putMetadata(Osx.Header.TargetNodeID.name(), "9999");
            inboundBuilder.putMetadata(Osx.Header.SourceInstID.name(), "");
            inboundBuilder.putMetadata(Osx.Header.TargetInstID.name(), "");
            inboundBuilder.putMetadata(Osx.Header.SessionID.name(), "testSessionID");
            inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), "PRODUCE_MSG");
            inboundBuilder.putMetadata(Osx.Metadata.TargetComponentName.name(), "");
            inboundBuilder.putMetadata(Osx.Metadata.SourceComponentName.name(), "");
            inboundBuilder.putMetadata(Osx.Metadata.MessageTopic.name(), transferId);
            inboundBuilder.putMetadata(Osx.Metadata.MessageCode.name(), Long.toString(System.currentTimeMillis()));
            //inboundBuilder.getMetadataMap().put(Pcp.Metadata.MessageOffSet.name(),);
            Osx.Message.Builder messageBuilder = Osx.Message.newBuilder();
            messageBuilder.setBody(ByteString.copyFrom(("test body element " + i).getBytes()));
            messageBuilder.setHead(ByteString.copyFrom(("test head " + i).getBytes()));
            inboundBuilder.setPayload(messageBuilder.build().toByteString());
            Osx.Outbound outbound = blockingStub.invoke(inboundBuilder.build());
            System.err.println("response " + outbound);
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    @Test
    public void  testTopicApply(){

        Osx.Inbound.Builder  inboundBuilder = Osx.Inbound.newBuilder();
        //  inboundBuilder.putMetadata(Osx.Header.Version.name(), "123");
        inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(),  MetaInfo.PROPERTY_FATE_TECH_PROVIDER );
        inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), TargetMethod.APPLY_TOPIC.name());
        inboundBuilder.putMetadata(Osx.Metadata.MessageTopic.name(), "testTopic0001");
        inboundBuilder.putMetadata(Osx.Metadata.InstanceId.name(),"localhost:9999" );
        inboundBuilder.putMetadata(Osx.Header.SessionID.name(), "testSessionId");


//        TopicApplyRequest  topicApplyRequest = new TopicApplyRequest();
//        topicApplyRequest.setTopic("testTopic0001");
//        topicApplyRequest.setInstanceId("localhost:9999");
//        topicApplyRequest.setSessionId("testSessionId");
      //  inboundBuilder.setPayload(ByteString.copyFrom(JsonUtil.object2Json(topicApplyRequest).getBytes(StandardCharsets.UTF_8)));
//        RouterInfo routerInfo = new  RouterInfo();
//        routerInfo.setHost("localhost");
//        routerInfo.setPort(9370);
//        ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo,true);
        // FireworkServiceGrpc.FireworkServiceBlockingStub stub = FireworkServiceGrpc.newBlockingStub(managedChannel);
        //PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub  stub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
        Osx.Outbound  outbound = blockingStub.invoke(inboundBuilder.build());

        //TopicApplyResponse  topicApplyResponse = JsonUtil.json2Object(outbound.getPayload().toByteArray(),TopicApplyResponse.class);

        System.err.println(outbound);

    }

    public void test07Ack(long index) {

        Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
        inboundBuilder.putMetadata(Osx.Header.Version.name(), "123");
        inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(),  MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        inboundBuilder.putMetadata(Osx.Header.Token.name(), "testToken");
        inboundBuilder.putMetadata(Osx.Header.SourceNodeID.name(), "9999");
        inboundBuilder.putMetadata(Osx.Header.TargetNodeID.name(), "10000");
        inboundBuilder.putMetadata(Osx.Header.SourceInstID.name(), "");
        inboundBuilder.putMetadata(Osx.Header.TargetInstID.name(), "");
        inboundBuilder.putMetadata(Osx.Header.SessionID.name(), "testSessionID");
        inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), TargetMethod.ACK_MSG.name());
        inboundBuilder.putMetadata(Osx.Metadata.TargetComponentName.name(), "");
        inboundBuilder.putMetadata(Osx.Metadata.SourceComponentName.name(), "");
        inboundBuilder.putMetadata(Osx.Metadata.MessageTopic.name(), transferId);
        inboundBuilder.putMetadata(Osx.Metadata.MessageOffSet.name(), Long.toString(index));
        Osx.Outbound outbound = blockingStub.invoke(inboundBuilder.build());
        System.err.println("ack response:" + outbound);
    }

    @Test
    public void test06UnaryConsume() {
        boolean needContinue = true;
        Osx.Outbound consumeResponse;
        int count = 0;
        do {
            System.err.println("===================");
            Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
            inboundBuilder.putMetadata(Osx.Header.Version.name(), "123");
            inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(),  MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
            inboundBuilder.putMetadata(Osx.Header.Token.name(), "testToken");
            inboundBuilder.putMetadata(Osx.Header.SourceNodeID.name(), "9999");
            inboundBuilder.putMetadata(Osx.Header.TargetNodeID.name(), "9999");
            inboundBuilder.putMetadata(Osx.Header.SourceInstID.name(), "");
            inboundBuilder.putMetadata(Osx.Header.TargetInstID.name(), "");
            inboundBuilder.putMetadata(Osx.Header.SessionID.name(), "testSessionID");
            inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), TargetMethod.CONSUME_MSG.name());
            inboundBuilder.putMetadata(Osx.Metadata.TargetComponentName.name(), "");
            inboundBuilder.putMetadata(Osx.Metadata.SourceComponentName.name(), "");
            inboundBuilder.putMetadata(Osx.Metadata.MessageTopic.name(), transferId);
            inboundBuilder.putMetadata(Osx.Metadata.MessageOffSet.name(), "-1");
            inboundBuilder.putMetadata(Osx.Metadata.Timeout.name(),"100000");
            consumeResponse = blockingStub.invoke(inboundBuilder.build());
            System.err.println("response : "+consumeResponse);

            String indexString = consumeResponse.getMetadataMap().get(Osx.Metadata.MessageOffSet.name());
            Long index = Long.parseLong(indexString);
            test07Ack(index);
            String code = consumeResponse.getCode();
            String msg = consumeResponse.getMessage();
            if (code.equals("0")) {
                try {
                    Osx.Message message = Osx.Message.parseFrom(consumeResponse.getPayload());
                } catch (InvalidProtocolBufferException e) {
                    e.printStackTrace();
                }
            }
            index++;
            count++;
        } while (count < 100);
    }

    @Test
    public void test07CancelTransfer() {

        Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
        inboundBuilder.putMetadata(Osx.Header.Version.name(), "123");
        inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(),  MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        inboundBuilder.putMetadata(Osx.Header.Token.name(), "testToken");
        inboundBuilder.putMetadata(Osx.Header.SourceNodeID.name(), "9999");
        inboundBuilder.putMetadata(Osx.Header.TargetNodeID.name(), "10000");
        inboundBuilder.putMetadata(Osx.Header.SourceInstID.name(), "");
        inboundBuilder.putMetadata(Osx.Header.TargetInstID.name(), "");
        inboundBuilder.putMetadata(Osx.Header.SessionID.name(), "testSessionID");
        inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), TargetMethod.CONSUME_MSG.name());
        inboundBuilder.putMetadata(Osx.Metadata.TargetComponentName.name(), "");
        inboundBuilder.putMetadata(Osx.Metadata.SourceComponentName.name(), "");
        inboundBuilder.putMetadata(Osx.Metadata.MessageTopic.name(), transferId);
        Osx.Outbound outbound = blockingStub.invoke(inboundBuilder.build());
        System.err.println("cancel result ：" + outbound);
    }


    public  static  void  main(String[] args){
        

    }

}
