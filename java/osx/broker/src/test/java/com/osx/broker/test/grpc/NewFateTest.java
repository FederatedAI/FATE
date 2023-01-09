package com.osx.broker.test.grpc;

import com.osx.core.constant.Dict;
import com.osx.core.ptp.TargetMethod;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContextBuilder;
import org.junit.Before;
import org.junit.Test;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;

import java.io.File;
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

    @Before
    public void init() {
       ManagedChannel managedChannel = createManagedChannel(ip, port);
        //  stub =      PrivateTransferProtocolGrpc.newBlockingStub();
       // ManagedChannel managedChannel2 = createManagedChannel(ip, port);
        blockingStub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
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
    public void testUnaryCall(){
        Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
        inboundBuilder.putMetadata(Osx.Header.Version.name(), "123");
        inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(), Dict.FATE_TECH_PROVIDER);
        inboundBuilder.putMetadata(Osx.Header.Token.name(), "testToken");
        inboundBuilder.putMetadata(Osx.Header.SourceNodeID.name(), "9999");
        inboundBuilder.putMetadata(Osx.Header.TargetNodeID.name(), "10000");
        inboundBuilder.putMetadata(Osx.Header.SourceInstID.name(), "");
        inboundBuilder.putMetadata(Osx.Header.TargetInstID.name(), "");
        inboundBuilder.putMetadata(Osx.Header.SessionID.name(), "testSessionID");
        inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), TargetMethod.UNARY_CALL.name());
        inboundBuilder.putMetadata(Osx.Metadata.TargetComponentName.name(), "fateflow");
        inboundBuilder.putMetadata(Osx.Metadata.SourceComponentName.name(), "");
        // inboundBuilder.putMetadata(Osx.Metadata.MessageTopic.name(), transferId);
        Osx.Outbound outbound = blockingStub.invoke(inboundBuilder.build());
        System.err.println("response : "+outbound);
    }

}
