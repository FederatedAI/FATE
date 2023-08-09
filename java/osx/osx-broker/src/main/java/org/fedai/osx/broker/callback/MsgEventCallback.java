package org.fedai.osx.broker.callback;

import org.fedai.osx.broker.message.MessageExt;
import org.fedai.osx.broker.queue.TransferQueue;

@FunctionalInterface
public interface MsgEventCallback {
    void callback(TransferQueue transferQueue , MessageExt message) throws Exception;
}
