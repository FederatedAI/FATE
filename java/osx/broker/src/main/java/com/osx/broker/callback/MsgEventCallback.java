package com.osx.broker.callback;

import com.osx.broker.message.Message;
import com.osx.broker.queue.TransferQueue;

@FunctionalInterface
public interface MsgEventCallback {
    void callback(TransferQueue transferQueue , Message message);
}
