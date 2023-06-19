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
package com.osx.broker.message;

import com.google.common.collect.Maps;
import com.osx.broker.constants.MessageFlag;
import com.osx.broker.util.MessageId;
import com.osx.broker.util.UtilAll;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.*;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MessageDecoder {


    static Logger logger = LoggerFactory.getLogger(MessageDecoder.class);

    public final static Charset CHARSET_UTF8 = Charset.forName("UTF-8");
    public final static int MESSAGE_MAGIC_CODE_POSTION = 4;
    public final static int MESSAGE_FLAG_POSTION = 16;
    public final static int MESSAGE_PHYSIC_OFFSET_POSTION = 28;
    //    public final static int MESSAGE_STORE_TIMESTAMP_POSTION = 56;
    public final static int MESSAGE_MAGIC_CODE = -626843481;
    public static final char NAME_VALUE_SEPARATOR = 1;
    public static final char PROPERTY_SEPARATOR = 2;
    public static final int PHY_POS_POSITION = 4 + 4 + 4 + 4 + 4 + 8;
    public static final int QUEUE_OFFSET_POSITION = 4 + 4 + 4 + 4 + 4;
    public static final int SYSFLAG_POSITION = 4 + 4 + 4 + 4 + 4 + 8 + 8;


    public static String createMessageId(final ByteBuffer input, final ByteBuffer addr, final long offset) {
        input.flip();
        int msgIDLength = addr.limit() == 8 ? 16 : 28;
        input.limit(msgIDLength);

        input.put(addr);
        input.putLong(offset);

        return UtilAll.bytes2string(input.array());
    }

    public static MessageExtBrokerInner buildMessageExtBrokerInner(String topic, byte[] body,
                                                                   String  msgCode, MessageFlag flag, String srcPartyId, String desPartyId) {
        MessageExtBrokerInner messageExtBrokerInner = new MessageExtBrokerInner();
        messageExtBrokerInner.setBody(body);
        messageExtBrokerInner.setTopic(topic);
        messageExtBrokerInner.setFlag(flag.getFlag());
        messageExtBrokerInner.setBornTimestamp(System.currentTimeMillis());
        messageExtBrokerInner.setDesPartyId(desPartyId);
        messageExtBrokerInner.setSrcPartyId(srcPartyId);
        messageExtBrokerInner.setProperties(Maps.newHashMap());
        messageExtBrokerInner.setMsgId(msgCode);
        return messageExtBrokerInner;
    }

//    public static String createMessageId(SocketAddress socketAddress, long transactionIdhashCode) {
//        InetSocketAddress inetSocketAddress = (InetSocketAddress) socketAddress;
//        int msgIDLength = inetSocketAddress.getAddress() instanceof Inet4Address ? 16 : 28;
//        ByteBuffer byteBuffer = ByteBuffer.allocate(msgIDLength);
//        byteBuffer.put(inetSocketAddress.getAddress().getAddress());
//        byteBuffer.putInt(inetSocketAddress.getPort());
//        byteBuffer.putLong(transactionIdhashCode);
//        byteBuffer.flip();
//        return UtilAll.bytes2string(byteBuffer.array());
//    }

//    public static MessageId decodeMessageId(final String msgId) throws UnknownHostException {
//        SocketAddress address;
//        long offset;
//        int ipLength = msgId.length() == 32 ? 4 * 2 : 16 * 2;
//
//        byte[] ip = UtilAll.string2bytes(msgId.substring(0, ipLength));
//        byte[] port = UtilAll.string2bytes(msgId.substring(ipLength, ipLength + 8));
//        ByteBuffer bb = ByteBuffer.wrap(port);
//        int portInt = bb.getInt(0);
//        address = new InetSocketAddress(InetAddress.getByAddress(ip), portInt);
//
//        // offset
//        byte[] data = UtilAll.string2bytes(msgId.substring(ipLength + 8, ipLength + 8 + 16));
//        bb = ByteBuffer.wrap(data);
//        offset = bb.getLong(0);
//
//        return new MessageId(address, offset);
//    }

    /**
     * Just decode properties from msg buffer.
     *
     * @param byteBuffer msg commit log buffer.
     */
//    public static Map<String, String> decodeProperties(ByteBuffer byteBuffer) {
//        int sysFlag = byteBuffer.getInt(SYSFLAG_POSITION);
//        int bornhostLength = (sysFlag & MessageSysFlag.BORNHOST_V6_FLAG) == 0 ? 8 : 20;
//        int storehostAddressLength = (sysFlag & MessageSysFlag.STOREHOSTADDRESS_V6_FLAG) == 0 ? 8 : 20;
//        int bodySizePosition = 4 // 1 TOTALSIZE
//                + 4 // 2 MAGICCODE
//                + 4 // 3 BODYCRC
//                + 4 // 4 QUEUEID
//                + 4 // 5 FLAG
//                + 8 // 6 QUEUEOFFSET
//                + 8 // 7 PHYSICALOFFSET
//                + 4 // 8 SYSFLAG
//                + 8 // 9 BORNTIMESTAMP
//                + bornhostLength // 10 BORNHOST
//                + 8 // 11 STORETIMESTAMP
//                + storehostAddressLength // 12 STOREHOSTADDRESS
//                + 4 // 13 RECONSUMETIMES
//                + 8; // 14 Prepared Transaction Offset
//        int topicLengthPosition = bodySizePosition + 4 + byteBuffer.getInt(bodySizePosition);
//
//        byte topicLength = byteBuffer.get(topicLengthPosition);
//
//        short propertiesLength = byteBuffer.getShort(topicLengthPosition + 1 + topicLength);
//
//        byteBuffer.position(topicLengthPosition + 1 + topicLength + 2);
//
//        if (propertiesLength > 0) {
//            byte[] properties = new byte[propertiesLength];
//            byteBuffer.get(properties);
//            String propertiesString = new String(properties, CHARSET_UTF8);
//            Map<String, String> map = string2messageProperties(propertiesString);
//            return map;
//        }
//        return null;
//    }

    public static MessageExt decode(ByteBuffer byteBuffer) {
        return decode(byteBuffer, true, true, false);
    }

//    public static MessageExt clientDecode(ByteBuffer byteBuffer, final boolean readBody) {
//        return decode(byteBuffer, readBody, true, true);
//    }

    public static MessageExt decode(ByteBuffer byteBuffer, final boolean readBody) {
        return decode(byteBuffer, readBody, true, false);
    }

    public static MessageExt decode(
            ByteBuffer byteBuffer, final boolean readBody, final boolean deCompressBody) {
        return decode(byteBuffer, readBody, deCompressBody, false);
    }

    public static MessageExt decode(
            ByteBuffer byteBuffer, final boolean readBody, final boolean deCompressBody, final boolean isClient) {
        try {

            MessageExt msgExt= new MessageExt();
            // 1 TOTALSIZE
            int storeSize = byteBuffer.getInt();
            msgExt.setStoreSize(storeSize);

//            // 2 MAGICCODE
//            byteBuffer.getInt();
//
//            // 3 BODYCRC
//            int bodyCRC = byteBuffer.getInt();
//            msgExt.setBodyCRC(bodyCRC);
//
//            // 4 QUEUEID
//            int queueId = byteBuffer.getInt();
//            msgExt.setQueueId(queueId);

            // 5 FLAG
            int flag = byteBuffer.getInt();
            msgExt.setFlag(flag);

            // 6 QUEUEOFFSET
            int srcPartyIdLength = byteBuffer.get();
            if (srcPartyIdLength > 0) {
                byte[] srcPartyBytes = new byte[srcPartyIdLength];
                byteBuffer.get(srcPartyBytes);
                String srcPartyId = new String(srcPartyBytes);
                msgExt.setSrcPartyId(srcPartyId);
            }

//            long queueOffset = byteBuffer.getLong();
//            msgExt.setQueueOffset(queueOffset);

            // 7 PHYSICALOFFSET
//            long physicOffset = byteBuffer.getLong();
//            msgExt.setCommitLogOffset(physicOffset);


            int desPartyIdLength = byteBuffer.get();
            if (desPartyIdLength > 0) {
                byte[] desPartyIdBytes = new byte[desPartyIdLength];
                byteBuffer.get(desPartyIdBytes);
                String desPartyId = new String(desPartyIdBytes);
                msgExt.setDesPartyId(desPartyId);
            }


            // 8 SYSFLAG
            int sysFlag = byteBuffer.getInt();
            msgExt.setSysFlag(sysFlag);

            // 9 BORNTIMESTAMP
            long bornTimeStamp = byteBuffer.getLong();
            msgExt.setBornTimestamp(bornTimeStamp);


            // 15 BODY
            int bodyLen = byteBuffer.getInt();
            if (bodyLen > 0) {
                if (readBody) {
                    byte[] body = new byte[bodyLen];
                    byteBuffer.get(body);
                    msgExt.setBody(body);
                } else {
                    byteBuffer.position(byteBuffer.position() + bodyLen);
                }
            }

            // 16 TOPIC
            short topicLen = byteBuffer.getShort();
            byte[] topic = new byte[(int) topicLen];
            byteBuffer.get(topic);
            msgExt.setTopic(new String(topic, CHARSET_UTF8));

            // 17 properties
            short propertiesLength = byteBuffer.getShort();

            if (propertiesLength > 0) {
                byte[] properties = new byte[propertiesLength];
                byteBuffer.get(properties);
                String propertiesString = new String(properties, CHARSET_UTF8);
                Map<String, String> map = string2messageProperties(propertiesString);
                msgExt.setProperties(map);

            }
            return msgExt;
        } catch (Exception e) {
            e.printStackTrace();
            byteBuffer.position(byteBuffer.limit());
        }

        return null;
    }

//    public static List<MessageExt> decodes(ByteBuffer byteBuffer) {
//        return decodes(byteBuffer, true);
//    }

//    public static List<MessageExt> decodes(ByteBuffer byteBuffer, final boolean readBody) {
//        List<MessageExt> msgExts = new ArrayList<MessageExt>();
//        while (byteBuffer.hasRemaining()) {
//            MessageExt msgExt = clientDecode(byteBuffer, readBody);
//            if (null != msgExt) {
//                msgExts.add(msgExt);
//            } else {
//                break;
//            }
//        }
//        return msgExts;
//    }

    public static String messageProperties2String(Map<String, String> properties) {
        StringBuilder sb = new StringBuilder();
        if (properties != null) {
            for (final Map.Entry<String, String> entry : properties.entrySet()) {
                final String name = entry.getKey();
                final String value = entry.getValue();

                if (value == null) {
                    continue;
                }
                sb.append(name);
                sb.append(NAME_VALUE_SEPARATOR);
                sb.append(value);
                sb.append(PROPERTY_SEPARATOR);
            }
        }
        return sb.toString();
    }

    public static Map<String, String> string2messageProperties(final String properties) {
        Map<String, String> map = new HashMap<String, String>();
        if (properties != null) {
            String[] items = properties.split(String.valueOf(PROPERTY_SEPARATOR));
            for (String i : items) {
                String[] nv = i.split(String.valueOf(NAME_VALUE_SEPARATOR));
                if (2 == nv.length) {
                    map.put(nv[0], nv[1]);
                }
            }
        }

        return map;
    }

//    public static byte[] encodeMessage(Message message) {
//        //only need flag, body, properties
//        byte[] body = message.getBody();
//        int bodyLen = body.length;
//        String properties = messageProperties2String(message.getProperties());
//        byte[] propertiesBytes = properties.getBytes(CHARSET_UTF8);
//        //note properties length must not more than Short.MAX
//        short propertiesLength = (short) propertiesBytes.length;
//        int sysFlag = message.getFlag();
//        int storeSize = 4 // 1 TOTALSIZE
//                + 4 // 2 MAGICCOD
//                + 4 // 3 BODYCRC
//                + 4 // 4 FLAG
//                + 4 + bodyLen // 4 BODY
//                + 2 + propertiesLength;
//        ByteBuffer byteBuffer = ByteBuffer.allocate(storeSize);
//        // 1 TOTALSIZE
//        byteBuffer.putInt(storeSize);
//
//        // 2 MAGICCODE
//        byteBuffer.putInt(0);
//
//        // 3 BODYCRC
//        byteBuffer.putInt(0);
//
//        // 4 FLAG
//        int flag = message.getFlag();
//        byteBuffer.putInt(flag);
//
//        // 5 BODY
//        byteBuffer.putInt(bodyLen);
//        byteBuffer.put(body);
//
//        // 6 properties
//        byteBuffer.putShort(propertiesLength);
//        byteBuffer.put(propertiesBytes);
//
//        return byteBuffer.array();
//    }

//    public static Message decodeMessage(ByteBuffer byteBuffer) throws Exception {
//        Message message = new Message();
//
//        // 1 TOTALSIZE
//        byteBuffer.getInt();
//
//        // 2 MAGICCODE
//        byteBuffer.getInt();
//
//        // 3 BODYCRC
//        byteBuffer.getInt();
//
//        // 4 FLAG
//        int flag = byteBuffer.getInt();
//        message.setFlag(flag);
//
//        // 5 BODY
//        int bodyLen = byteBuffer.getInt();
//        byte[] body = new byte[bodyLen];
//        byteBuffer.get(body);
//        message.setBody(body);
//
//        // 6 properties
//        short propertiesLen = byteBuffer.getShort();
//        byte[] propertiesBytes = new byte[propertiesLen];
//        byteBuffer.get(propertiesBytes);
//        message.setProperties(string2messageProperties(new String(propertiesBytes, CHARSET_UTF8)));
//
//        return message;
//    }

//    public static byte[] encodeMessages(List<Message> messages) {
//        //TO DO refactor, accumulate in one buffer, avoid copies
//        List<byte[]> encodedMessages = new ArrayList<byte[]>(messages.size());
//        int allSize = 0;
//        for (Message message : messages) {
//            byte[] tmp = encodeMessage(message);
//            encodedMessages.add(tmp);
//            allSize += tmp.length;
//        }
//        byte[] allBytes = new byte[allSize];
//        int pos = 0;
//        for (byte[] bytes : encodedMessages) {
//            System.arraycopy(bytes, 0, allBytes, pos, bytes.length);
//            pos += bytes.length;
//        }
//        return allBytes;
//    }

//    public static List<Message> decodeMessages(ByteBuffer byteBuffer) throws Exception {
//        //TO DO add a callback for processing,  avoid creating lists
//        List<Message> msgs = new ArrayList<Message>();
//        while (byteBuffer.hasRemaining()) {
//            Message msg = decodeMessage(byteBuffer);
//            msgs.add(msg);
//        }
//        return msgs;
//    }
}
