package com.osx.broker.callback;

import com.osx.broker.message.Message;
import com.osx.broker.message.MessageExt;
import com.osx.broker.queue.TransferQueue;

@FunctionalInterface
public interface MsgEventCallback {
    void callback(TransferQueue transferQueue , MessageExt message) throws Exception;
}
