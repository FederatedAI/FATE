package com.osx.broker.grpc;

import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;

public class PullRequestDataWrap {

    Proxy.Metadata metadata;
    StreamObserver streamObserver;

    public Proxy.Metadata getMetadata() {
        return metadata;
    }

    public void setMetadata(Proxy.Metadata metadata) {
        this.metadata = metadata;
    }

    public StreamObserver getStreamObserver() {
        return streamObserver;
    }

    public void setStreamObserver(StreamObserver streamObserver) {
        this.streamObserver = streamObserver;
    }

}
