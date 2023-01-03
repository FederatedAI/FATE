package com.osx.broker.grpc;

public class TransferConf {

    int maxConcurrentCallPerConnection = 10;
    //= CoreConfKeys.CONFKEY_CORE_GRPC_SERVER_CHANNEL_MAX_CONCURRENT_CALL_PER_CONNECTION.getWith(options).toInt
    int maxInboundMessageSize = 3888888;
    //= CoreConfKeys.CONFKEY_CORE_GRPC_SERVER_CHANNEL_MAX_INBOUND_MESSAGE_SIZE.getWith(options).toInt
    int maxInboundMetadataSize = 3888883;
    //        = CoreConfKeys.CONFKEY_CORE_GRPC_SERVER_CHANNEL_MAX_INBOUND_METADATA_SIZE.getWith(options).toInt
    int flowControlWindow = 100;
    //= CoreConfKeys.CONFKEY_CORE_GRPC_SERVER_CHANNEL_FLOW_CONTROL_WINDOW.getWith(options).toInt
    int channelKeepAliveTimeSec = 10;
    //= CoreConfKeys.CONFKEY_CORE_GRPC_SERVER_CHANNEL_KEEPALIVE_TIME_SEC.getWith(options).toLong
    int channelKeepAliveTimeoutSec = 10;
    //= CoreConfKeys.CONFKEY_CORE_GRPC_SERVER_CHANNEL_KEEPALIVE_TIMEOUT_SEC.getWith(options).toLong
    int channelPermitKeepAliveTime = 100;
    //= CoreConfKeys.CONFKEY_CORE_GRPC_SERVER_CHANNEL_PERMIT_KEEPALIVE_TIME_SEC.getWith(options).toLong
    int channelKeepAliveWithoutCallsEnabled;
    //= CoreConfKeys.CONFKEY_CORE_GRPC_SERVER_CHANNEL_KEEPALIVE_WITHOUT_CALLS_ENABLED.getWith(options).toBoolean
    long maxConnectionIdle = 10;
    //= CoreConfKeys.CONFKEY_CORE_GRPC_SERVER_CHANNEL_MAX_CONNECTION_IDLE_SEC.getWith(options).toLong
    long maxConnectionAge = 10;
    //= CoreConfKeys.CONFKEY_CORE_GRPC_SERVER_CHANNEL_MAX_CONNECTION_AGE_SEC.getWith(options).toLong
    long maxConnectionAgeGrace = 100;
    //= CoreConfKeys.CONFKEY_CORE_GRPC_SERVER_CHANNEL_MAX_CONNECTION_AGE_GRACE_SEC.getWith(options).toLong
    int port = 9002;
    String host = "localhost";

    public int getMaxConcurrentCallPerConnection() {
        return maxConcurrentCallPerConnection;
    }

    public void setMaxConcurrentCallPerConnection(int maxConcurrentCallPerConnection) {
        this.maxConcurrentCallPerConnection = maxConcurrentCallPerConnection;
    }

    public int getMaxInboundMessageSize() {
        return maxInboundMessageSize;
    }

    public void setMaxInboundMessageSize(int maxInboundMessageSize) {
        this.maxInboundMessageSize = maxInboundMessageSize;
    }

    public int getMaxInboundMetadataSize() {
        return maxInboundMetadataSize;
    }

    public void setMaxInboundMetadataSize(int maxInboundMetadataSize) {
        this.maxInboundMetadataSize = maxInboundMetadataSize;
    }

    public int getFlowControlWindow() {
        return flowControlWindow;
    }

    public void setFlowControlWindow(int flowControlWindow) {
        this.flowControlWindow = flowControlWindow;
    }

    public int getChannelKeepAliveTimeSec() {
        return channelKeepAliveTimeSec;
    }

    public void setChannelKeepAliveTimeSec(int channelKeepAliveTimeSec) {
        this.channelKeepAliveTimeSec = channelKeepAliveTimeSec;
    }

    public int getChannelKeepAliveTimeoutSec() {
        return channelKeepAliveTimeoutSec;
    }

    public void setChannelKeepAliveTimeoutSec(int channelKeepAliveTimeoutSec) {
        this.channelKeepAliveTimeoutSec = channelKeepAliveTimeoutSec;
    }

    public int getChannelPermitKeepAliveTime() {
        return channelPermitKeepAliveTime;
    }

    public void setChannelPermitKeepAliveTime(int channelPermitKeepAliveTime) {
        this.channelPermitKeepAliveTime = channelPermitKeepAliveTime;
    }

    public int getChannelKeepAliveWithoutCallsEnabled() {
        return channelKeepAliveWithoutCallsEnabled;
    }

    public void setChannelKeepAliveWithoutCallsEnabled(int channelKeepAliveWithoutCallsEnabled) {
        this.channelKeepAliveWithoutCallsEnabled = channelKeepAliveWithoutCallsEnabled;
    }

    public long getMaxConnectionIdle() {
        return maxConnectionIdle;
    }

    public void setMaxConnectionIdle(long maxConnectionIdle) {
        this.maxConnectionIdle = maxConnectionIdle;
    }

    public long getMaxConnectionAge() {
        return maxConnectionAge;
    }

    public void setMaxConnectionAge(long maxConnectionAge) {
        this.maxConnectionAge = maxConnectionAge;
    }

    public long getMaxConnectionAgeGrace() {
        return maxConnectionAgeGrace;
    }

    public void setMaxConnectionAgeGrace(long maxConnectionAgeGrace) {
        this.maxConnectionAgeGrace = maxConnectionAgeGrace;
    }

    public int getPort() {
        return port;
    }

    public void setPort(int port) {
        this.port = port;
    }

    public String getHost() {
        return host;
    }

    public void setHost(String host) {
        this.host = host;
    }

}
