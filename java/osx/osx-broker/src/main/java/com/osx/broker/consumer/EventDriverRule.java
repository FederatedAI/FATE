package com.osx.broker.consumer;

import com.osx.broker.queue.TransferQueue;
@FunctionalInterface
public interface EventDriverRule {
    boolean isMatch(TransferQueue queue);
}
