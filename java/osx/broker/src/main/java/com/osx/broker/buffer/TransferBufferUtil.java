package com.osx.broker.buffer;

import java.nio.ByteBuffer;

public class TransferBufferUtil {

    public static byte[] conver(ByteBuffer byteBuffer) {
        int len = byteBuffer.limit() - byteBuffer.position();
        byte[] bytes = new byte[len];

        if (byteBuffer.isReadOnly()) {
            return null;
        } else {
            byteBuffer.get(bytes);
            return bytes;
        }
    }
}
