package org.fedai.osx.broker.test.grpc;
//import com.firework.cluster.rpc.FireworkQueueServiceGrpc;
//import com.firework.cluster.rpc.FireworkTransfer;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;

import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.context.Protocol;
import org.fedai.osx.core.ptp.TargetMethod;
import org.fedai.osx.core.router.RouterInfo;
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
    static String ip = "localhost";
    //int port = 8250;//nginx
    static int port = 9377;//nginx
    static String desPartyId = "9999";
    static String desRole = "";
    static String srcPartyId = "10000";
    static String srcRole = "";
    static String sessionId = "testSessionId";
    static String topic ="testTopic";
    static OsxContext  fateContext= new OsxContext();
    static RouterInfo routerInfo= new RouterInfo();
    static {
        routerInfo.setHost(ip);
        routerInfo.setPort(port);
        routerInfo.setProtocol(Protocol.grpc);
        routerInfo.setUrl("http://localhost:8087/osx/inbound");
        //HttpClientPool.initPool();
    }

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
    }

    private  byte[] createBigArray(int size){
        byte[]  result = new  byte[size];
        for(int i=0;i<size;i++){
            result[i]=1;
        }
        return  result;
    }


    @Test
    public void testPeek(){
        //202310191251296433960_toy_example_0_0.202310191251296433960_toy_example_0_0-host-10000-guest-9999-host_index
        //202310191310469345390_toy_example_0_0.202310191310469345390_toy_example_0_0-host-10000-guest-9999-host_index
        Osx.PeekInbound.Builder inboundBuilder = Osx.PeekInbound.newBuilder();
        inboundBuilder.setTopic("202310191310469345390_toy_example_0_0-host-10000-guest-9999-host_index");
        OsxContext  fateContext= new OsxContext();
        fateContext.setTraceId(Long.toString(System.currentTimeMillis()));
        fateContext.setSessionId("202310191310469345390_toy_example_0_0");
//        fateContext.setTopic("test_topic");
        //fateContext.setDesInstId("webank");
//        fateContext.setDesNodeId("10000");
        fateContext.setUri(UriConstants.PEEK);
        fateContext.setTechProviderCode(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        OsxContext.pushThreadLocalContext(fateContext);
        Osx.TransportOutbound outbound =TransferUtil.redirectPeek(fateContext,routerInfo,inboundBuilder.build());
        System.err.println("result "+  outbound);
    }




    @Test
    public void testPop(){

        Osx.PopInbound.Builder inboundBuilder = Osx.PopInbound.newBuilder();
        inboundBuilder.setTopic(topic);
        OsxContext  fateContext= new OsxContext();
        fateContext.setTraceId(Long.toString(System.currentTimeMillis()));
        fateContext.setSessionId(sessionId);
        fateContext.setUri(UriConstants.POP);
        fateContext.setTechProviderCode(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        OsxContext.pushThreadLocalContext(fateContext);
        Osx.TransportOutbound outbound =TransferUtil.redirectPop(fateContext,routerInfo,inboundBuilder.build());
        System.err.println("result : "+  new String(outbound.getPayload().toByteArray()));
    }

        @Test
        public void testPush() {
        for (int i = 0; i < 1; i++) {

            Osx.PushInbound.Builder pushInbound = Osx.PushInbound.newBuilder();
            pushInbound.setTopic(topic);
            pushInbound.setPayload(ByteString.copyFrom(("my name is "+i).getBytes(StandardCharsets.UTF_8)));
            OsxContext  fateContext= new OsxContext();
            fateContext.setTraceId("fate-test-"+System.currentTimeMillis());
            fateContext.setSessionId(sessionId);
            fateContext.setTopic(topic);
            fateContext.setDesNodeId(desPartyId);
            fateContext.setTechProviderCode(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
            OsxContext.pushThreadLocalContext(fateContext);
            Osx.TransportOutbound outbound =TransferUtil.redirectPush(fateContext,pushInbound.build(),routerInfo,true);
            System.err.println("response " + outbound);
        }
    }


//    @Test
//    public void  testTopicApply(){
//        Osx.Inbound.Builder  inboundBuilder = Osx.Inbound.newBuilder();
//        //  inboundBuilder.putMetadata(Osx.Header.Version.name(), "123");
//        inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(),  MetaInfo.PROPERTY_FATE_TECH_PROVIDER );
//        inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), TargetMethod.APPLY_TOPIC.name());
//        inboundBuilder.putMetadata(Osx.Metadata.MessageTopic.name(), "testTopic0001");
//        inboundBuilder.putMetadata(Osx.Metadata.InstanceId.name(),"localhost:9999" );
//        inboundBuilder.putMetadata(Osx.Header.SessionID.name(), "testSessionId");
//        Osx.Outbound outbound =TransferUtil.redirect(fateContext,inboundBuilder.build(),routerInfo,false);
//        System.err.println(outbound);
//    }

    @Test
    public void testRelease() {
        Osx.ReleaseInbound.Builder  releaseInboundBuilder = Osx.ReleaseInbound.newBuilder();
        releaseInboundBuilder.setTopic(topic);
        OsxContext  fateContext= new OsxContext();
        fateContext.setSessionId(sessionId);
        fateContext.setTechProviderCode(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        Osx.ReleaseInbound releaseInbound=   releaseInboundBuilder.build();
        OsxContext.pushThreadLocalContext(fateContext);
        Osx.TransportOutbound  outbound = TransferUtil.redirectRelease(fateContext,routerInfo,releaseInbound);
        System.err.println(outbound);
    }


    public  static  void  main(String[] args){
        

    }

}
