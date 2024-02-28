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
package org.fedai.osx.broker.eggroll;

import com.google.protobuf.AbstractMessageLite;
import com.webank.eggroll.core.command.Command;
import com.webank.eggroll.core.command.CommandServiceGrpc;
import io.grpc.ManagedChannel;
import org.fedai.osx.core.frame.GrpcConnectionFactory;
import org.fedai.osx.core.router.RouterInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.stream.Collectors;

public class CommandClient {

    Logger logger = LoggerFactory.getLogger(CommandClient.class);
    ErEndpoint erEndpoint;

    ManagedChannel managedChannel;

    public CommandClient(ErEndpoint erEndpoint) {
        this.erEndpoint = erEndpoint;
    }

    private synchronized ManagedChannel buildManagedChannel(String ip, int port) {
        if (managedChannel == null) {
//            NettyChannelBuilder channelBuilder = NettyChannelBuilder
//                    .forAddress(ip, port)
//                .keepAliveTime(60, TimeUnit.MINUTES)
//                .keepAliveTimeout(60, TimeUnit.MINUTES)
//                .keepAliveWithoutCalls(true)
////                    .idleTimeout(120, TimeUnit.SECONDS)
//                    .perRpcBufferLimit(128 << 20)
//                    .flowControlWindow(32 << 20)
//                    .maxInboundMessageSize(32 << 20)
//                    .enableRetry()
//                    .retryBufferSize(16 << 20)
//                    .maxRetryAttempts(20);
//            channelBuilder.usePlaintext();
//            managedChannel = channelBuilder.build();
            RouterInfo   routerInfo = new RouterInfo();
            routerInfo.setHost(ip);
            routerInfo.setPort(port);

            managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo);
        }
        return managedChannel;
    }


    public Command.CommandResponse call(CommandURI commandUri, BaseProto... baseProtos) {

        String id = System.currentTimeMillis() + "_" + commandUri.uri.toString();
        Command.CommandRequest commandRequest = Command.CommandRequest.newBuilder()
                .setId(id)
                .setUri(commandUri.uri.toString())
                .addAllArgs(Arrays.stream(baseProtos).
                        map((element) -> ((AbstractMessageLite) element.toProto()).toByteString()).collect(Collectors.toList()))
                .build();

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
