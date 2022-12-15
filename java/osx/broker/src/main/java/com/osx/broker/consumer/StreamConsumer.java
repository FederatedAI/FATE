package com.osx.broker.consumer;

public class StreamConsumer extends  LocalQueueConsumer{


    public StreamConsumer(long consumerId, String transferId) {
        super(consumerId, transferId);
    }
}
