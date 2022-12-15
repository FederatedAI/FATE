
package com.osx.broker.message;

import java.nio.ByteBuffer;



public interface AppendMessageHandler {


    AppendMessageResult doAppend(final long fileFromOffset, final ByteBuffer byteBuffer,
                                 final int maxBlank, final MessageExtBrokerInner msg);


}
