package org.fedai.osx.broker.test.grpc;

import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.context.Protocol;
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
    static String ip = "localhost";
    static int port = 9377;
    static String desPartyId = "10000";
    static String desRole = "";
    static String srcPartyId = "10000";
    static String srcRole = "";
    static String sessionId = "testSessionId";
    static String topic = "testTopic";
    static OsxContext fateContext = new OsxContext();
    static RouterInfo routerInfo = new RouterInfo();

    static {
        routerInfo.setHost(ip);
        routerInfo.setPort(port);
        routerInfo.setProtocol(Protocol.grpc);
//        routerInfo.setUseSSL(true);
        routerInfo.setUseKeyStore(false);
//        routerInfo.setTrustStorePassword("123456");
//        routerInfo.setTrustStoreFilePath("D:\\\\webank\\\\osx\\\\test2\\\\client_truststore.jks");
//        routerInfo.setKeyStorePassword("123456");
//        routerInfo.setKeyStoreFilePath("D:\\\\webank\\\\osx\\\\test2\\\\client.jks");

        routerInfo.setCertChainFile("/Users/kaideng/work/cert/test/client.crt");
        routerInfo.setCaFile("/Users/kaideng/work/cert/test/ca.crt");
        routerInfo.setPrivateKeyFile("/Users/kaideng/work/cert/test/client.pem");


        routerInfo.setTrustStorePassword("123456");
        routerInfo.setTrustStoreFilePath("D:\\\\webank\\\\osx\\\\test3\\\\client\\\\truststore.jks");
        routerInfo.setKeyStorePassword("123456");
        routerInfo.setKeyStoreFilePath("D:\\\\webank\\\\osx\\\\test3\\\\client\\\\identity.jks");
//        routerInfo.setNegotiationType("TLS");

        //  routerInfo.setUrl("http://localhost:8087/osx/inbound");
        //HttpClientPool.initPool();
    }
//    static {
//        routerInfo.setHost(ip);
//        routerInfo.setPort(9884);
//        routerInfo.setProtocol(Protocol.grpc);
////        keyStoreFilePath": "D:\\webank\\osx\\test\\server.keystore",
////        "keyStorePassword": "123456",
////                "trustStoreFilePath":"D:\\webank\\osx\\test\\server.keystore",
////                "trustStorePassword": "123456",
////                "negotiationType": "TLS",
////                "useSSL": true,
//        routerInfo.setUseSSL(true);
//        routerInfo.setUseKeyStore(true);
//        routerInfo.setTrustStorePassword("123456");
//        routerInfo.setTrustStoreFilePath("D:\\webank\\\\osx\\test\\server.keystore");
//        routerInfo.setKeyStorePassword("123456");
//        routerInfo.setKeyStoreFilePath("D:\\webank\\osx\\test\\server.keystore");
//        routerInfo.setNegotiationType("TLS");
//
//      //  routerInfo.setUrl("http://localhost:8087/osx/inbound");
//        //HttpClientPool.initPool();
//    }

    Logger logger = LoggerFactory.getLogger(QueueTest.class);
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

    public static void main(String[] args) {


    }

    @Before
    public void init() {
    }

    private byte[] createBigArray(int size) {
        byte[] result = new byte[size];
        for (int i = 0; i < size; i++) {
            result[i] = 1;
        }
        return result;
    }

    @Test
    public void testPeek() {
        Osx.PeekInbound.Builder inboundBuilder = Osx.PeekInbound.newBuilder();
        inboundBuilder.setTopic(topic);
        OsxContext fateContext = new OsxContext();
        fateContext.setTraceId(Long.toString(System.currentTimeMillis()));
        fateContext.setSessionId(sessionId);
//        fateContext.setTopic("test_topic");
        //fateContext.setDesInstId("webank");
//        fateContext.setDesNodeId("10000");
        fateContext.setUri(UriConstants.PEEK);
        fateContext.setTechProviderCode(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        OsxContext.pushThreadLocalContext(fateContext);
        Osx.TransportOutbound outbound = TransferUtil.redirectPeek(fateContext, routerInfo, inboundBuilder.build());
        System.err.println("result " + outbound);
    }

    @Test
    public void testPop() {

        Osx.PopInbound.Builder inboundBuilder = Osx.PopInbound.newBuilder();
        inboundBuilder.setTopic(topic);
        OsxContext fateContext = new OsxContext();
        fateContext.setTraceId(Long.toString(System.currentTimeMillis()));
        fateContext.setSessionId(sessionId);
        fateContext.setUri(UriConstants.POP);
        fateContext.setTechProviderCode(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        OsxContext.pushThreadLocalContext(fateContext);
        Osx.TransportOutbound outbound = TransferUtil.redirectPop(fateContext, routerInfo, inboundBuilder.build());
        System.err.println("result : " + new String(outbound.getPayload().toByteArray()));
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
    public void testPush() {
        for (int i = 0; i < 1; i++) {

            Osx.PushInbound.Builder pushInbound = Osx.PushInbound.newBuilder();
            pushInbound.setTopic(topic);
            pushInbound.setPayload(ByteString.copyFrom(("my name is " + i).getBytes(StandardCharsets.UTF_8)));
            OsxContext fateContext = new OsxContext();
            fateContext.setTraceId("fate-test-" + System.currentTimeMillis());
            fateContext.setSessionId(sessionId);
            fateContext.setTopic(topic);
            fateContext.setDesNodeId("10000");
            fateContext.setTechProviderCode(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
            OsxContext.pushThreadLocalContext(fateContext);
            Osx.TransportOutbound outbound = TransferUtil.redirectPush(fateContext, pushInbound.build(), routerInfo, true);
            System.err.println("response " + outbound);
        }
    }

    @Test
    public void testRelease() {
        Osx.ReleaseInbound.Builder releaseInboundBuilder = Osx.ReleaseInbound.newBuilder();
        releaseInboundBuilder.setTopic(topic);
        OsxContext fateContext = new OsxContext();
        fateContext.setSessionId(sessionId);
        fateContext.setTechProviderCode(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        Osx.ReleaseInbound releaseInbound = releaseInboundBuilder.build();
        OsxContext.pushThreadLocalContext(fateContext);
        Osx.TransportOutbound outbound = TransferUtil.redirectRelease(fateContext, routerInfo, releaseInbound);
        System.err.println(outbound);
    }

    @Test
    public void testInvoke() {
        Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
        inboundBuilder.setPayload(ByteString.copyFrom("hello world".getBytes(StandardCharsets.UTF_8)));
        OsxContext fateContext = new OsxContext();
        fateContext.setProtocol(Protocol.grpc);
        fateContext.setSessionId(sessionId);
        fateContext.setTechProviderCode(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        OsxContext.pushThreadLocalContext(fateContext);
        Object result = TransferUtil.redirect(fateContext, inboundBuilder.build(), routerInfo, true);
        System.err.println("result:" + result);

    }


    @Test
    public  void testUnaryCall() {

        ManagedChannel managedChannel = createManagedChannel(ip, port);
        DataTransferServiceGrpc.DataTransferServiceBlockingStub stub = DataTransferServiceGrpc.newBlockingStub(managedChannel);
        Proxy.Packet.Builder builder = Proxy.Packet.newBuilder();
        Transfer.RollSiteHeader.Builder headerBuilder = Transfer.RollSiteHeader.newBuilder();
        headerBuilder.setDstPartyId(desPartyId);
        builder.setHeader(Proxy.Metadata.newBuilder().setExt(headerBuilder.build().toByteString()));
        Proxy.Data.Builder dataBuilder = Proxy.Data.newBuilder();
        dataBuilder.setKey("name");
        dataBuilder.setValue(ByteString.copyFrom(("xiaoxiao").getBytes()));
        builder.setBody(Proxy.Data.newBuilder().setValue(ByteString.copyFromUtf8("hello world")));
        Proxy.Packet result = stub.unaryCall(builder.build());
        System.err.println(result);
    }

}
