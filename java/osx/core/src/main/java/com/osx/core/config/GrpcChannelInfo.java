package com.osx.core.config;

import io.grpc.netty.shaded.io.grpc.netty.NegotiationType;
import lombok.Data;

import java.util.concurrent.TimeUnit;

@Data
public class GrpcChannelInfo {

    int   keepAliveTime;
    boolean  keepAliveWithoutCalls= true;
    int   idelTimeOut;
    int   perRpcBufferLimit;
    int   flowControlWindow;
    int   maxInboundMessageSize;
    int   retryBufferSize;
    int   maxRetryAttemps;
}
