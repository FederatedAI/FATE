package com.osx.broker.message;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;

public class DefaultAppendMessageHandler implements AppendMessageHandler {

        Logger log = LoggerFactory.getLogger(DefaultAppendMessageHandler.class);
        // File at the end of the minimum fixed length empty
        private static final int END_FILE_MIN_BLANK_LENGTH = 4 + 4;
        private final ByteBuffer msgIdMemory;
        private final ByteBuffer msgIdV6Memory;
        // Store the message content
        private final ByteBuffer msgStoreItemMemory;
        // The maximum length of the message
        private final int maxMessageSize;
        // Build Message Key
        private final StringBuilder keyBuilder = new StringBuilder();

        private final StringBuilder msgIdBuilder = new StringBuilder();

        public DefaultAppendMessageHandler(final int size) {
            this.msgIdMemory = ByteBuffer.allocate(4 + 4 + 8);
            this.msgIdV6Memory = ByteBuffer.allocate(16 + 4 + 8);
            log.info("======================{}",size);
            this.msgStoreItemMemory = ByteBuffer.allocate(size + END_FILE_MIN_BLANK_LENGTH);
            this.maxMessageSize = size;
        }

        public ByteBuffer getMsgStoreItemMemory() {
            return msgStoreItemMemory;
        }

        public AppendMessageResult doAppend(final long fileFromOffset, final ByteBuffer byteBuffer, final int maxBlank,
                                            final MessageExtBrokerInner msgInner) {


            long wroteOffset = fileFromOffset + byteBuffer.position();
            String  msgId =  Long.toString(wroteOffset);
            Long queueOffset=new Long(0);
            final byte[] propertiesData =
                msgInner.getPropertiesString() == null ? null : msgInner.getPropertiesString().getBytes(MessageDecoder.CHARSET_UTF8);

            final int propertiesLength = propertiesData == null ? 0 : propertiesData.length;

            if (propertiesLength > Short.MAX_VALUE) {
                log.warn("putMessage message properties length too long. length={}", propertiesData.length);
                return new AppendMessageResult(AppendMessageStatus.PROPERTIES_SIZE_EXCEEDED);
            }

            final byte[] topicData = msgInner.getTopic().getBytes(MessageDecoder.CHARSET_UTF8);

            final byte[] srcPartyId =
                    msgInner.getSrcPartyId() == null ? null : msgInner.getSrcPartyId().getBytes(MessageDecoder.CHARSET_UTF8);
            final int srcPartyIdLength =  srcPartyId!=null?srcPartyId.length:0;

            final byte[] desPartyId =
                    msgInner.getDesPartyId() == null ? null : msgInner.getDesPartyId().getBytes(MessageDecoder.CHARSET_UTF8);
            final int desPartyIdLength =  desPartyId!=null?desPartyId.length:0;


            final int topicLength = topicData.length;

            final int bodyLength = msgInner.getBody() == null ? 0 : msgInner.getBody().length;

            final int msgLen = calMsgLength(msgInner.getSysFlag(),srcPartyIdLength,desPartyIdLength, bodyLength, topicLength, propertiesLength);

            // Exceeds the maximum message
            if (msgLen > this.maxMessageSize) {
                return new AppendMessageResult(AppendMessageStatus.MESSAGE_SIZE_EXCEEDED);
            }

            // Determines whether there is sufficient free space
            if ((msgLen + END_FILE_MIN_BLANK_LENGTH) > maxBlank) {
                this.resetByteBuffer(this.msgStoreItemMemory, maxBlank);
                // 1 TOTALSIZE
                this.msgStoreItemMemory.putInt(maxBlank);
//                // 2 MAGICCODE
//                this.msgStoreItemMemory.putInt(1111);
                // 3 The remaining space may be any value
                // Here the length of the specially set maxBlank
                final long beginTimeMills = System.currentTimeMillis();
                byteBuffer.put(this.msgStoreItemMemory.array(), 0, maxBlank);
                return new AppendMessageResult(AppendMessageStatus.END_OF_FILE, wroteOffset, maxBlank, msgId, msgInner.getStoreTimestamp(),
                    queueOffset, System.currentTimeMillis() - beginTimeMills);
            }

            // Initialization of storage space
            this.resetByteBuffer(msgStoreItemMemory, msgLen);

            // 1 TOTALSIZE
            this.msgStoreItemMemory.putInt(msgLen);
          //  log.info("msgLen {}",msgLen);
            // 2 MAGICCODE
            this.msgStoreItemMemory.putInt(1000);
            // 3 BODYCRC
            this.msgStoreItemMemory.putInt(msgInner.getBodyCRC());
            // 4 QUEUEID
            this.msgStoreItemMemory.putInt(msgInner.getQueueId());
            // 5 FLAG
            this.msgStoreItemMemory.putInt(msgInner.getFlag());
            // 6 QUEUEOFFSET

            this.msgStoreItemMemory.put((byte)srcPartyIdLength);
            if(srcPartyId!=null)
                this.msgStoreItemMemory.put(srcPartyId);

            this.msgStoreItemMemory.put((byte)desPartyIdLength);
            if(desPartyId!=null)
                this.msgStoreItemMemory.put(desPartyId);

           // this.msgStoreItemMemory.putLong(fileFromOffset + byteBuffer.position());
            // 8 SYSFLAG
            this.msgStoreItemMemory.putInt(msgInner.getSysFlag());
            // 9 BORNTIMESTAMP
            this.msgStoreItemMemory.putLong(msgInner.getBornTimestamp());
//            // 10 BORNHOST
//            this.resetByteBuffer(bornHostHolder, bornHostLength);
//            this.msgStoreItemMemory.put(msgInner.getBornHostBytes(bornHostHolder));
//            // 11 STORETIMESTAMP
//            this.msgStoreItemMemory.putLong(msgInner.getStoreTimestamp());
//            // 12 STOREHOSTADDRESS
//            this.resetByteBuffer(storeHostHolder, storeHostLength);
//            this.msgStoreItemMemory.put(msgInner.getStoreHostBytes(storeHostHolder));
//            // 13 RECONSUMETIMES
//            this.msgStoreItemMemory.putInt(msgInner.getReconsumeTimes());
//            // 14 Prepared Transaction Offset
//            this.msgStoreItemMemory.putLong(msgInner.getPreparedTransactionOffset());
            // 15 BODY
            this.msgStoreItemMemory.putInt(bodyLength);
            if (bodyLength > 0)
                this.msgStoreItemMemory.put(msgInner.getBody());
            // 16 TOPIC
            this.msgStoreItemMemory.putShort((short) topicLength);
            this.msgStoreItemMemory.put(topicData);
            // 17 PROPERTIES
            this.msgStoreItemMemory.putShort((short) propertiesLength);
            if (propertiesLength > 0)
                this.msgStoreItemMemory.put(propertiesData);

            final long beginTimeMills = System.currentTimeMillis();
            // Write messages to the queue buffer
            byteBuffer.put(this.msgStoreItemMemory.array(), 0, msgLen);

            AppendMessageResult result = new AppendMessageResult(AppendMessageStatus.PUT_OK, wroteOffset, msgLen, msgId,
                msgInner.getStoreTimestamp(), queueOffset,  0);
            return result;
        }



        private void resetByteBuffer(final ByteBuffer byteBuffer, final int limit) {
            byteBuffer.flip();
            byteBuffer.limit(limit);
        }

    protected static int calMsgLength(int sysFlag, int srcPartyIdLength ,int desPartyIdLength,int bodyLength, int topicLength, int propertiesLength) {
        int bornhostLength = (sysFlag & MessageSysFlag.BORNHOST_V6_FLAG) == 0 ? 8 : 20;
        int storehostAddressLength = (sysFlag & MessageSysFlag.STOREHOSTADDRESS_V6_FLAG) == 0 ? 8 : 20;
        final int msgLen = 4 //TOTALSIZE
                + 4 //MAGICCODE
                + 4 //BODYCRC
                + 4 //QUEUEID
                + 4 //FLAG
                //+ 8 //QUEUEOFFSET
                + 1 +  (srcPartyIdLength > 0 ? srcPartyIdLength : 0)

               // + 8 //PHYSICALOFFSET
                + 1 +  (desPartyIdLength > 0 ? desPartyIdLength : 0)
                + 4 //SYSFLAG
                + 8 //BORNTIMESTAMP
               // + bornhostLength //BORNHOST
               // + 8 //STORETIMESTAMP
               // + storehostAddressLength //STOREHOSTADDRESS
               // + 4 //RECONSUMETIMES
              //  + 8 //Prepared Transaction Offset
                + 4 + (bodyLength > 0 ? bodyLength : 0) //BODY
                + 2 + topicLength //TOPIC
                + 2 + (propertiesLength > 0 ? propertiesLength : 0) //propertiesLength
                + 0;
        return msgLen;
    }


    }