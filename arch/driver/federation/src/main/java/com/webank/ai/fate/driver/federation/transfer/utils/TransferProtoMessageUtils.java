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

package com.webank.ai.fate.driver.federation.transfer.utils;

import com.google.common.base.Preconditions;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.eggroll.core.constant.StringConstants;
import org.apache.commons.lang3.SerializationException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

@Component
@Scope("prototype")
// todo: merge proto messages
public class TransferProtoMessageUtils {
    public static Proxy.Command COMMAND_SEND_START = Proxy.Command.newBuilder().setName(StringConstants.SEND_START).build();
    public static Proxy.Command COMMAND_SEND_END = Proxy.Command.newBuilder().setName(StringConstants.SEND_END).build();
    private final Object builderLock = new Object();
    @Autowired
    private TransferPojoUtils transferPojoUtils;
    private Proxy.Packet.Builder packetBuilder;
    private Proxy.Metadata.Builder headerBuilder;
    private Proxy.Data.Builder bodyBuilder;
    private Proxy.Topic.Builder topicBuilder;
    private Proxy.Task.Builder taskBuilder;
    private volatile boolean inited = false;

    private synchronized void init() {
        if (inited) {
            return;
        }
        this.packetBuilder = Proxy.Packet.newBuilder();
        this.headerBuilder = Proxy.Metadata.newBuilder();
        this.bodyBuilder = Proxy.Data.newBuilder();
        this.topicBuilder = Proxy.Topic.newBuilder();
        this.taskBuilder = Proxy.Task.newBuilder();

        inited = true;
    }

    public Proxy.Topic partyToTopic(Federation.Party party) {
        if (!inited) {
            init();
        }
        return topicBuilder.clear()
                .setPartyId(party.getPartyId())
                .setName(party.getName())
                .setRole("fate")
                .build();
    }

    public Federation.TransferMeta extractTransferMetaFromPacket(Proxy.Packet packet) {
        ByteString serialized = packet.getBody().getValue();

        Preconditions.checkNotNull(serialized, "null serialized transferMeta");
        try {
            return Federation.TransferMeta.parseFrom(serialized);
        } catch (InvalidProtocolBufferException e) {
            throw new SerializationException(e);
        }
    }

    public Proxy.Packet generateSendStartRequest(Federation.TransferMeta transferMeta) {
        return generateSendCommandRequest(transferMeta, COMMAND_SEND_START);
    }

    public Proxy.Packet generateSendEndRequest(Federation.TransferMeta transferMeta) {
        return generateSendCommandRequest(transferMeta, COMMAND_SEND_END);
    }

    private Proxy.Packet generateSendCommandRequest(Federation.TransferMeta transferMeta, Proxy.Command command) {
        if (!inited) {
            init();
        }

        packetBuilder.clear();
        headerBuilder.clear();
        bodyBuilder.clear();

        headerBuilder
                .setTask(taskBuilder.clear().setTaskId(transferPojoUtils.generateTransferId(transferMeta)))
                .setSrc(this.partyToTopic(transferMeta.getSrc()))
                .setDst(this.partyToTopic(transferMeta.getDst()))
                .setCommand(command);

        bodyBuilder.setValue(transferMeta.toByteString());

        packetBuilder.setHeader(headerBuilder);
        packetBuilder.setBody(bodyBuilder);

        return packetBuilder.build();
    }
}
