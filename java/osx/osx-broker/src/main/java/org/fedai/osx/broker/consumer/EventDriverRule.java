package org.fedai.osx.broker.consumer;

import org.fedai.osx.broker.queue.AbstractQueue;

@FunctionalInterface
public interface EventDriverRule {
    boolean isMatch(AbstractQueue queue);
}
