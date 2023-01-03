package com.osx.broker.eggroll;

import com.google.protobuf.AbstractMessageLite;
import com.webank.eggroll.core.command.Command;
import com.webank.eggroll.core.command.CommandServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class CommandClient {

    Logger logger = LoggerFactory.getLogger(CommandClient.class);
    ErEndpoint erEndpoint;

    public CommandClient(ErEndpoint erEndpoint) {
        this.erEndpoint = erEndpoint;
    }

    private ManagedChannel buildManagedChannel(String ip, int port) {
        NettyChannelBuilder channelBuilder = NettyChannelBuilder
                .forAddress(ip, port)
                .keepAliveTime(60, TimeUnit.MINUTES)
                .keepAliveTimeout(60, TimeUnit.MINUTES)
                .keepAliveWithoutCalls(true)
                .idleTimeout(60, TimeUnit.MINUTES)
                .perRpcBufferLimit(128 << 20)
                .flowControlWindow(32 << 20)
                .maxInboundMessageSize(32 << 20)
                .enableRetry()
                .retryBufferSize(16 << 20)
                .maxRetryAttempts(20);
        channelBuilder.usePlaintext();

        return channelBuilder.build();

    }


    public Command.CommandResponse call(CommandURI commandUri, BaseProto... baseProtos) {

        String id = System.currentTimeMillis() + "_" + commandUri.uri.toString();
        Command.CommandRequest commandRequest = Command.CommandRequest.newBuilder()
                .setId(id)
                .setUri(commandUri.uri.toString())
                .addAllArgs(Arrays.stream(baseProtos).
                        map((element) -> ((AbstractMessageLite) element.toProto()).toByteString()).collect(Collectors.toList()))
                .build();
        logger.info("===call {} {} id {}", erEndpoint.host, erEndpoint.port, id);

        ManagedChannel managedChannel = buildManagedChannel(erEndpoint.host, erEndpoint.port);
        CommandServiceGrpc.CommandServiceBlockingStub stub = CommandServiceGrpc.newBlockingStub(managedChannel);
        Command.CommandResponse commandResponse = stub.call(commandRequest);
        return commandResponse;
    }

//    def call[T](commandUri: CommandURI, args: RpcMessage*)(implicit tag:ClassTag[T]): T = {
//        logDebug(s"[CommandClient.call, single endpoint] commandUri: ${commandUri.uriString}, endpoint: ${defaultEndpoint}")
//        try {
//            val stub = CommandServiceGrpc.newBlockingStub(GrpcClientUtils.getChannel(defaultEndpoint))
//            val argBytes = args.map(x => ByteString.copyFrom(SerdesUtils.rpcMessageToBytes(x, SerdesTypes.PROTOBUF)))
//            val resp = stub.call(Command.CommandRequest.newBuilder
//                    .setId(System.currentTimeMillis + "")
//                    .setUri(commandUri.uri.toString)
//                    .addAllArgs(argBytes.asJava)
//                    .build)
//            SerdesUtils.rpcMessageFromBytes(resp.getResults(0).toByteArray,
//                    tag.runtimeClass, SerdesTypes.PROTOBUF).asInstanceOf[T]
//        } catch {
//            case t: Throwable =>
//                logError(s"[COMMAND] error calling to ${defaultEndpoint}, message: ${args(0)}. commandUri: ${commandUri.uriString}", t)
//                throw new CommandCallException(commandUri, defaultEndpoint, t)
//        }
//    }

}
