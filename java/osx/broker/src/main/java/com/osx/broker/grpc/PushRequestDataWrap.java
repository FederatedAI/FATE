package com.osx.broker.grpc;

import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;

public class PushRequestDataWrap {
    Proxy.Packet  packet;
    StreamObserver   streamObserver;

    public Proxy.Packet getPacket() {
        return packet;
    }

    public void setPacket(Proxy.Packet packet) {
        this.packet = packet;
    }

    public StreamObserver getStreamObserver() {
        return streamObserver;
    }

    public void setStreamObserver(StreamObserver streamObserver) {
        this.streamObserver = streamObserver;
    }
}
