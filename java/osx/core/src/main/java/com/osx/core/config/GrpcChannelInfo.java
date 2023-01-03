package com.osx.core.config;

import lombok.Data;

@Data
public class GrpcChannelInfo {

    int keepAliveTime;
    int keepAliveTimeout;
    boolean keepAliveWithoutCalls = true;
    int idelTimeOut;
    int perRpcBufferLimit;
    int flowControlWindow;
    int maxInboundMessageSize;
    int retryBufferSize;
    int maxRetryAttemps;


}
