package com.osx.broker.test.grpc;
//import com.firework.cluster.rpc.FireworkQueueServiceGrpc;
//import com.firework.cluster.rpc.FireworkTransfer;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.osx.core.ptp.TargetMethod;
import com.osx.federation.rpc.Osx;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.stub.StreamObserver;
import  org.junit.After;
import org.junit.FixMethodOrder;
import  org.junit.Test;
import  org.junit.Before;
import org.junit.runners.MethodSorters;
import org.ppc.ptp.Pcp;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.*;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class QueueTest {
    Logger logger = LoggerFactory.getLogger(QueueTest.class);
    String  ip=  "localhost";
    //int port = 8250;//nginx
    int port = 9889;//nginx
    String desPartyId = "10000";
    String desRole = "";
    String srcPartyId = "9999";
    String srcRole = "";
    String transferId = "testTransferId";
    String sessionId =  "testSessionId";




    PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub blockingStub;
//    FireworkQueueServiceGrpc.FireworkQueueServiceBlockingStub  blockingStub;

    @Before
    public  void init (){
        ManagedChannel managedChannel = createManagedChannel(ip,port);
      //  stub =      PrivateTransferProtocolGrpc.newBlockingStub();
        ManagedChannel managedChannel2 = createManagedChannel(ip,port);
        blockingStub =     PrivateTransferProtocolGrpc.newBlockingStub(managedChannel2);
    }





    @Test
    public  void test02Query() {
        Pcp.Inbound.Builder  inboundBuilder = Pcp.Inbound.newBuilder();
        inboundBuilder.putMetadata(Pcp.Header.Version.name(),"123");
        inboundBuilder.putMetadata(Pcp.Header.TechProviderCode.name(),"FT");
        inboundBuilder.putMetadata(Pcp.Header.Token.name(),"testToken");
        inboundBuilder.putMetadata(Pcp.Header.SourceNodeID.name(),"9999");
        inboundBuilder.putMetadata(Pcp.Header.TargetNodeID.name(),"10000");
        inboundBuilder.putMetadata(Pcp.Header.SourceInstID.name(),"");
        inboundBuilder.putMetadata(Pcp.Header.TargetInstID.name(),"");
        inboundBuilder.putMetadata(Pcp.Header.SessionID.name(),"testSessionID");
        inboundBuilder.putMetadata(Pcp.Metadata.TargetMethod.name(), TargetMethod.QUERY_TOPIC.name());
        inboundBuilder.putMetadata(Pcp.Metadata.TargetComponentName.name(),"");
        inboundBuilder.putMetadata(Pcp.Metadata.SourceComponentName.name(),"");
        inboundBuilder.putMetadata(Pcp.Metadata.MessageTopic.name(),transferId);


        Pcp.Outbound outbound = blockingStub.invoke(inboundBuilder.build());
        Osx.TopicInfo topicInfo= null;
        try {
            topicInfo = Osx.TopicInfo.parseFrom(outbound.getPayload());
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }
        System.err.println("response " +topicInfo);
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }





//    @Test
//    public void  test03Consume(){
//        System.err.println("testConsume");
//        CountDownLatch countDownLatch = new CountDownLatch(1);
//
//        FireworkTransfer.ConsumeRequest consumeRequest =  FireworkTransfer.ConsumeRequest.newBuilder().setTransferId(this.transferId).build();
//
//        stub.consume(consumeRequest, new StreamObserver<FireworkTransfer.ConsumeResponse>() {
//
//            int  count =0;
//
//            @Override
//            public void onNext(FireworkTransfer.ConsumeResponse value) {
//                count++;
//                System.err.println("receive : "+value);
//            }
//
//            @Override
//            public void onError(Throwable t) {
//                t.printStackTrace();
//                System.err.println("error");
//                countDownLatch.countDown();
//            }
//
//            @Override
//            public void onCompleted() {
//                System.err.println("onCompleted");
//                countDownLatch.countDown();
//            }
//        });
//        try {
//            countDownLatch.await();
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
//
//
//    }


    @Test
    public  void  test04UnaryProduce(){
        for (int i = 0; i < 10; i++) {
            Pcp.Inbound.Builder  inboundBuilder = Pcp.Inbound.newBuilder();
            inboundBuilder.putMetadata(Pcp.Header.Version.name(),"123");
            inboundBuilder.putMetadata(Pcp.Header.TechProviderCode.name(),"FT");
            inboundBuilder.putMetadata(Pcp.Header.Token.name(),"testToken");
            inboundBuilder.putMetadata(Pcp.Header.SourceNodeID.name(),"9999");
            inboundBuilder.putMetadata(Pcp.Header.TargetNodeID.name(),"10000");
            inboundBuilder.putMetadata(Pcp.Header.SourceInstID.name(),"");
            inboundBuilder.putMetadata(Pcp.Header.TargetInstID.name(),"");
            inboundBuilder.putMetadata(Pcp.Header.SessionID.name(),"testSessionID");
            inboundBuilder.putMetadata(Pcp.Metadata.TargetMethod.name(),"PRODUCE_MSG");
            inboundBuilder.putMetadata(Pcp.Metadata.TargetComponentName.name(),"");
            inboundBuilder.putMetadata(Pcp.Metadata.SourceComponentName.name(),"");
            inboundBuilder.putMetadata(Pcp.Metadata.MessageTopic.name(),transferId);
            //inboundBuilder.getMetadataMap().put(Pcp.Metadata.MessageOffSet.name(),);
            Osx.Message.Builder  messageBuilder = Osx.Message.newBuilder();
            messageBuilder.setBody(ByteString.copyFrom(("test body element "+i).getBytes()));
            messageBuilder.setHead(ByteString.copyFrom(("test head "+i).getBytes()));
            inboundBuilder.setPayload(messageBuilder.build().toByteString());
            Pcp.Outbound outbound = blockingStub.invoke(inboundBuilder.build());
            System.err.println("response " +outbound);
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }






    public void test07Ack(long index){

        Pcp.Inbound.Builder  inboundBuilder = Pcp.Inbound.newBuilder();
        inboundBuilder.putMetadata(Pcp.Header.Version.name(),"123");
        inboundBuilder.putMetadata(Pcp.Header.TechProviderCode.name(),"FT");
        inboundBuilder.putMetadata(Pcp.Header.Token.name(),"testToken");
        inboundBuilder.putMetadata(Pcp.Header.SourceNodeID.name(),"9999");
        inboundBuilder.putMetadata(Pcp.Header.TargetNodeID.name(),"10000");
        inboundBuilder.putMetadata(Pcp.Header.SourceInstID.name(),"");
        inboundBuilder.putMetadata(Pcp.Header.TargetInstID.name(),"");
        inboundBuilder.putMetadata(Pcp.Header.SessionID.name(),"testSessionID");
        inboundBuilder.putMetadata(Pcp.Metadata.TargetMethod.name(),TargetMethod.ACK_MSG.name());
        inboundBuilder.putMetadata(Pcp.Metadata.TargetComponentName.name(),"");
        inboundBuilder.putMetadata(Pcp.Metadata.SourceComponentName.name(),"");
        inboundBuilder.putMetadata(Pcp.Metadata.MessageTopic.name(),transferId);
        inboundBuilder.putMetadata(Pcp.Metadata.MessageOffSet.name(), Long.toString(index));
        Pcp.Outbound   outbound =  blockingStub.invoke(inboundBuilder.build());
        System.err.println("ack response:"+outbound);
    }




    @Test
    public  void  test06UnaryConsume(){
        boolean needContinue= true;
        Pcp.Outbound consumeResponse;
        int count=0;
        do {
            System.err.println("===================");
            Pcp.Inbound.Builder  inboundBuilder = Pcp.Inbound.newBuilder();
            inboundBuilder.putMetadata(Pcp.Header.Version.name(),"123");
            inboundBuilder.putMetadata(Pcp.Header.TechProviderCode.name(),"FT");
            inboundBuilder.putMetadata(Pcp.Header.Token.name(),"testToken");
            inboundBuilder.putMetadata(Pcp.Header.SourceNodeID.name(),"9999");
            inboundBuilder.putMetadata(Pcp.Header.TargetNodeID.name(),"10000");
            inboundBuilder.putMetadata(Pcp.Header.SourceInstID.name(),"");
            inboundBuilder.putMetadata(Pcp.Header.TargetInstID.name(),"");
            inboundBuilder.putMetadata(Pcp.Header.SessionID.name(),"testSessionID");
            inboundBuilder.putMetadata(Pcp.Metadata.TargetMethod.name(),TargetMethod.CONSUME_MSG.name());
            inboundBuilder.putMetadata(Pcp.Metadata.TargetComponentName.name(),"");
            inboundBuilder.putMetadata(Pcp.Metadata.SourceComponentName.name(),"");
            inboundBuilder.putMetadata(Pcp.Metadata.MessageTopic.name(),transferId);
            inboundBuilder.putMetadata(Pcp.Metadata.MessageOffSet.name(), "-1");
            consumeResponse = blockingStub.invoke(inboundBuilder.build());
            System.err.println(consumeResponse);

            String indexString = consumeResponse.getMetadataMap().get(Pcp.Metadata.MessageOffSet.name());
            Long index =  Long.parseLong(indexString);
            test07Ack(index);
            String  code = consumeResponse.getCode();
            String  msg =  consumeResponse.getMessage();
            if(code.equals("0")) {
                try {
                    Osx.Message  message = Osx.Message.parseFrom(consumeResponse.getPayload());
                } catch (InvalidProtocolBufferException e) {
                    e.printStackTrace();
                }
            }
            index ++;
            count++;
        }while(count<10);
    }


    @Test
    public  void  test07CancelTransfer(){

        Pcp.Inbound.Builder  inboundBuilder = Pcp.Inbound.newBuilder();
        inboundBuilder.putMetadata(Pcp.Header.Version.name(),"123");
        inboundBuilder.putMetadata(Pcp.Header.TechProviderCode.name(),"FT");
        inboundBuilder.putMetadata(Pcp.Header.Token.name(),"testToken");
        inboundBuilder.putMetadata(Pcp.Header.SourceNodeID.name(),"9999");
        inboundBuilder.putMetadata(Pcp.Header.TargetNodeID.name(),"10000");
        inboundBuilder.putMetadata(Pcp.Header.SourceInstID.name(),"");
        inboundBuilder.putMetadata(Pcp.Header.TargetInstID.name(),"");
        inboundBuilder.putMetadata(Pcp.Header.SessionID.name(),"testSessionID");
        inboundBuilder.putMetadata(Pcp.Metadata.TargetMethod.name(),TargetMethod.CONSUME_MSG.name());
        inboundBuilder.putMetadata(Pcp.Metadata.TargetComponentName.name(),"");
        inboundBuilder.putMetadata(Pcp.Metadata.SourceComponentName.name(),"");
        inboundBuilder.putMetadata(Pcp.Metadata.MessageTopic.name(),transferId);
        Pcp.Outbound  outbound  = blockingStub.invoke(inboundBuilder.build());
        System.err.println("cancel result ："+outbound);
    }


    public static ManagedChannel createManagedChannel(String ip, int port) {
        try {
            NettyChannelBuilder channelBuilder = NettyChannelBuilder
                    .forAddress(ip, port)
                    .keepAliveTime(12, TimeUnit.SECONDS)
                    .keepAliveTimeout(1, TimeUnit.SECONDS)
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
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

}
