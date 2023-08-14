package org.fedai.osx.broker.test.grpc;

import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.stub.StreamObserver;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.ptp.TargetMethod;
import org.junit.Before;
import org.junit.Test;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;

import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class NewFateTest {

    String ip = "localhost";
    //int port = 8250;//nginx
    int port = 9370;//nginx
    String desPartyId = "10000";
    String desRole = "";
    String srcPartyId = "9999";
    String srcRole = "";
    String transferId = "testTransferId";
    String sessionId = "testSessionId";
    PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub  blockingStub;
    PrivateTransferProtocolGrpc.PrivateTransferProtocolStub stub;
    @Before
    public void init() {
       ManagedChannel managedChannel = createManagedChannel(ip, port);
        //  stub =      PrivateTransferProtocolGrpc.newBlockingStub();
       // ManagedChannel managedChannel2 = createManagedChannel(ip, port);
        blockingStub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
        stub = PrivateTransferProtocolGrpc.newStub(managedChannel);


    }

    public static ManagedChannel createManagedChannel(String ip, int port) {
        try {
            NettyChannelBuilder channelBuilder = NettyChannelBuilder
                    .forAddress(ip, port)
                    .keepAliveTime(60, TimeUnit.SECONDS)
                    .keepAliveTimeout(60, TimeUnit.SECONDS)
                    .keepAliveWithoutCalls(true)
                    .idleTimeout(60, TimeUnit.SECONDS)
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
            //      logger.error("create channel error : " ,e);
            //e.printStackTrace();
        }
        return null;
    }

    @Test
    public void testUnaryCall(byte[] data){
        Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
        inboundBuilder.putMetadata(Osx.Header.Version.name(), "123");
        inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(), "FATE");//
        inboundBuilder.putMetadata(Osx.Header.Token.name(), "testToken");
        inboundBuilder.putMetadata(Osx.Header.SourceNodeID.name(), "9999");
        inboundBuilder.putMetadata(Osx.Header.TargetNodeID.name(), "10000");
        inboundBuilder.putMetadata(Osx.Header.SourceInstID.name(), "9999");
        inboundBuilder.putMetadata(Osx.Header.TargetInstID.name(), "10000");
        inboundBuilder.putMetadata(Osx.Header.SessionID.name(), "testSessionID");
        inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), "UNARY_CALL");
        inboundBuilder.putMetadata(Osx.Metadata.TargetComponentName.name(), "fateflow");
        inboundBuilder.putMetadata(Osx.Metadata.SourceComponentName.name(), "");
        inboundBuilder.putMetadata(Osx.Header.TraceID.name(), "28938999993");
        inboundBuilder.setPayload(ByteString.copyFrom(data));
        Osx.Outbound outbound = blockingStub.invoke(inboundBuilder.build());
        System.err.println("response : "+outbound);
    }


    @Test
    public void testStream(){

        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.Inbound>  reqSb = stub.transport(new StreamObserver<Osx.Outbound>() {
            @Override
            public void onNext(Osx.Outbound outbound) {
                System.err.println(outbound);
            }
            @Override
            public void onError(Throwable throwable) {
                throwable.printStackTrace();
            }
            @Override
            public void onCompleted() {
                System.err.println("completed");
            }
        });
        for(int i=0;i<3;i++){
            Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
            inboundBuilder.putMetadata(Osx.Header.Version.name(), "123");
            inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(), MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
            inboundBuilder.putMetadata(Osx.Header.Token.name(), "testToken");
            inboundBuilder.putMetadata(Osx.Header.SourceNodeID.name(), "9999");
            inboundBuilder.putMetadata(Osx.Header.TargetNodeID.name(), "10000");
            inboundBuilder.putMetadata(Osx.Header.SourceInstID.name(), "");
            inboundBuilder.putMetadata(Osx.Header.TargetInstID.name(), "");
            inboundBuilder.putMetadata(Osx.Header.SessionID.name(), "testSessionID");
            inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), TargetMethod.TEST_STREAM.name());
            inboundBuilder.putMetadata(Osx.Metadata.TargetComponentName.name(), "");
            inboundBuilder.putMetadata(Osx.Metadata.SourceComponentName.name(), "");
            // inboundBuilder.putMetadata(Osx.Metadata.MessageTopic.name(), transferId);

            inboundBuilder.setPayload(ByteString.copyFrom(("test "+i).getBytes(StandardCharsets.UTF_8)));
            reqSb.onNext(inboundBuilder.build());
        }

        System.err.println("==========================");

    }


    public static void main(String[] args) {
        System.err.println("===============");
        NewFateTest  newFateTest = new  NewFateTest();
        newFateTest.init();
        newFateTest.testStream();

        CountDownLatch countDownLatch = new CountDownLatch(1);
        try {
            countDownLatch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }

}
