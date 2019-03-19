package com.webank.ai.fate.core.network.grpc.client;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
public class ChannelPool {
    private static final ManagedChannel proxyChannel = ManagedChannelBuilder.forAddress("127.0.0.1", 50052).usePlaintext().build();

    public ManagedChannel getProxyChannel(){
        return ChannelPool.proxyChannel;
    }

}
