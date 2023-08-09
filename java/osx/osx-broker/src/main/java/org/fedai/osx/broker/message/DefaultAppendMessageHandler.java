/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.fedai.osx.broker.message;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

public class DefaultAppendMessageHandler implements AppendMessageHandler {

    // File at the end of the minimum fixed length empty
    private static final int END_FILE_MIN_BLANK_LENGTH = 4 + 4;
    // Store the message content
    private final ByteBuffer msgStoreItemMemory;
    // The maximum length of the message
    private final int maxMessageSize;
    Logger log = LoggerFactory.getLogger(DefaultAppendMessageHandler.class);

    public DefaultAppendMessageHandler(final int size) {

        this.msgStoreItemMemory = ByteBuffer.allocate(size + END_FILE_MIN_BLANK_LENGTH);
        this.maxMessageSize = size;
    }
    protected static int calMsgLength(int sysFlag, int srcPartyIdLength, int desPartyIdLength, int bodyLength, int topicLength, int propertiesLength) {
        final int msgLen = 4 //TOTALSIZE
                + 4 //FLAG
                + 1 + (srcPartyIdLength > 0 ? srcPartyIdLength : 0)
                + 1 + (desPartyIdLength > 0 ? desPartyIdLength : 0)
                + 4 //SYSFLAG
                + 8 //BORNTIMESTAMP
                + 4 + (bodyLength > 0 ? bodyLength : 0) //BODY
                + 2 + topicLength //TOPIC
                + 2 + (propertiesLength > 0 ? propertiesLength : 0) //propertiesLength
                + 0;
        return msgLen;
    }

    public ByteBuffer getMsgStoreItemMemory() {
        return msgStoreItemMemory;
    }

    public AppendMessageResult doAppend(final long fileFromOffset, final ByteBuffer byteBuffer, final int maxBlank,
                                        final MessageExtBrokerInner msgInner) {


        long wroteOffset = fileFromOffset + byteBuffer.position();
        String msgId = Long.toString(wroteOffset);
        Long queueOffset = new Long(0);
        final byte[] propertiesData =
                msgInner.getProperties()==null  ? null : MessageDecoder.messageProperties2String (msgInner.getProperties()).getBytes(StandardCharsets.UTF_8);

        final int propertiesLength = propertiesData==null? 0 : propertiesData.length;
        if (propertiesLength > Short.MAX_VALUE) {
            log.warn("putMessage message properties length too long. length={}", propertiesData.length);
            return new AppendMessageResult(AppendMessageStatus.PROPERTIES_SIZE_EXCEEDED);
        }
        final byte[] topicData = msgInner.getTopic().getBytes(MessageDecoder.CHARSET_UTF8);
        final byte[] srcPartyId = msgInner.getSrcPartyId() == null ? null : msgInner.getSrcPartyId().getBytes(MessageDecoder.CHARSET_UTF8);
        final int srcPartyIdLength = srcPartyId != null ? srcPartyId.length : 0;
        final byte[] desPartyId = msgInner.getDesPartyId() == null ? null : msgInner.getDesPartyId().getBytes(MessageDecoder.CHARSET_UTF8);
        final int desPartyIdLength = desPartyId != null ? desPartyId.length : 0;
        final int topicLength = topicData.length;
        final int bodyLength = msgInner.getBody() == null ? 0 : msgInner.getBody().length;
        final int msgLen = calMsgLength(msgInner.getSysFlag(), srcPartyIdLength, desPartyIdLength, bodyLength, topicLength, propertiesLength);
        // Exceeds the maximum message
        if (msgLen > this.maxMessageSize) {
            return new AppendMessageResult(AppendMessageStatus.MESSAGE_SIZE_EXCEEDED);
        }
        // Determines whether there is sufficient free space
        if ((msgLen + END_FILE_MIN_BLANK_LENGTH) > maxBlank) {
            this.resetByteBuffer(this.msgStoreItemMemory, maxBlank);
            // 1 TOTALSIZE
            this.msgStoreItemMemory.putInt(maxBlank);
            final long beginTimeMills = System.currentTimeMillis();
            byteBuffer.put(this.msgStoreItemMemory.array(), 0, maxBlank);
            return new AppendMessageResult(AppendMessageStatus.END_OF_FILE, wroteOffset, maxBlank, msgId, msgInner.getStoreTimestamp(),
                    queueOffset, System.currentTimeMillis() - beginTimeMills);
        }
        // Initialization of storage space
        this.resetByteBuffer(msgStoreItemMemory, msgLen);
        // 1 TOTALSIZE
        this.msgStoreItemMemory.putInt(msgLen);
        // 5 FLAG
        this.msgStoreItemMemory.putInt(msgInner.getFlag());
        // 6 QUEUEOFFSET
        this.msgStoreItemMemory.put((byte) srcPartyIdLength);
        if (srcPartyId != null)
            this.msgStoreItemMemory.put(srcPartyId);
        this.msgStoreItemMemory.put((byte) desPartyIdLength);
        if (desPartyId != null)
            this.msgStoreItemMemory.put(desPartyId);
        // 8 SYSFLAG
        this.msgStoreItemMemory.putInt(msgInner.getSysFlag());
        // 9 BORNTIMESTAMP
        this.msgStoreItemMemory.putLong(msgInner.getBornTimestamp());
        this.msgStoreItemMemory.putInt(bodyLength);
        if (bodyLength > 0)
            this.msgStoreItemMemory.put(msgInner.getBody());
        // 16 TOPIC
        this.msgStoreItemMemory.putShort((short) topicLength);
        this.msgStoreItemMemory.put(topicData);
        // 17 PROPERTIES
        this.msgStoreItemMemory.putShort((short) propertiesLength);
        if (propertiesLength > 0) {
            this.msgStoreItemMemory.put(propertiesData);
        }
        byteBuffer.put(this.msgStoreItemMemory.array(), 0, msgLen);
        AppendMessageResult result = new AppendMessageResult(AppendMessageStatus.PUT_OK, wroteOffset, msgLen, msgId,
                msgInner.getStoreTimestamp(), queueOffset, 0);
        return result;
    }

    private void resetByteBuffer(final ByteBuffer byteBuffer, final int limit) {
        byteBuffer.flip();
        byteBuffer.limit(limit);
    }
}