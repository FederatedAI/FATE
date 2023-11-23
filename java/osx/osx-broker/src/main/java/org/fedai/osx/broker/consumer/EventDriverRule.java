package org.fedai.osx.broker.consumer;

import org.fedai.osx.broker.queue.AbstractQueue;
import org.fedai.osx.broker.queue.TransferQueue;

@FunctionalInterface
public interface EventDriverRule {
    boolean isMatch(AbstractQueue queue);
}
