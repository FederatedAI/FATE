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

package com.webank.ai.fate.networking.proxy.model;

import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.networking.proxy.event.model.PipeHandleNotificationEvent;
import com.webank.ai.fate.networking.proxy.infra.Pipe;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

@Component
@Scope("prototype")
public class PipeHandlerInfo {
    private PipeHandleNotificationEvent.Type type;
    private Proxy.Metadata metadata;
    private Pipe pipe;
    private Proxy.Packet packet;

    public PipeHandlerInfo(PipeHandleNotificationEvent.Type type, Proxy.Metadata metadata, Pipe pipe) {
        this.type = type;
        this.metadata = metadata;
        this.pipe = pipe;
    }

    public PipeHandlerInfo(PipeHandleNotificationEvent.Type type, Proxy.Packet packet, Pipe pipe) {
        this.type = type;
        this.packet = packet;
        this.metadata = packet.getHeader();
        this.pipe = pipe;
    }

    public PipeHandleNotificationEvent.Type getType() {
        return type;
    }

    public void setType(PipeHandleNotificationEvent.Type type) {
        this.type = type;
    }

    public Proxy.Metadata getMetadata() {
        return metadata;
    }

    public void setMetadata(Proxy.Metadata metadata) {
        this.metadata = metadata;
    }

    public Pipe getPipe() {
        return pipe;
    }

    public void setPipe(Pipe pipe) {
        this.pipe = pipe;
    }

    public Proxy.Packet getPacket() {
        return packet;
    }

    public void setPacket(Proxy.Packet packet) {
        this.packet = packet;
    }

    @Override
    public String toString() {
        return "PipeHandlerInfo{" +
                "type=" + type +
                ", metadata=" + metadata +
                ", pipe=" + pipe +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof PipeHandlerInfo)) return false;

        PipeHandlerInfo that = (PipeHandlerInfo) o;

        if (type != that.type) return false;
        if (!metadata.equals(that.metadata)) return false;
        return pipe.equals(that.pipe);
    }

    @Override
    public int hashCode() {
        int result = type.hashCode();
        result = 31 * result + metadata.hashCode();
        result = 31 * result + pipe.hashCode();
        return result;
    }
}
