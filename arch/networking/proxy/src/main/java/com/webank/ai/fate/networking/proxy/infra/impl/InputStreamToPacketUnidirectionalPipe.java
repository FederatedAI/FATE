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

package com.webank.ai.fate.networking.proxy.infra.impl;

import com.google.common.io.ByteStreams;
import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import org.apache.commons.io.IOUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.TimeUnit;

@Component
@Scope("prototype")
public class InputStreamToPacketUnidirectionalPipe extends BasePipe {
    private static final int MAX_EMPTY_READ_COUNT = 10;
    private static final int MAX_READ_COUNT = 50;
    private static final int DEFAULT_TRUNK_SIZE = 2 << 20;
    private static final Logger LOGGER = LogManager.getLogger(InputStreamToPacketUnidirectionalPipe.class);
    private final int trunkSize;
    private InputStream is;
    private Proxy.Metadata metadata;
    private Proxy.Packet.Builder packetBuilder;
    private Proxy.Data.Builder dataBuilder;
    private boolean hasReadBefore;
    private int counter = 0;

    public InputStreamToPacketUnidirectionalPipe(InputStream is, Proxy.Metadata metadata) {
        this(is, metadata, DEFAULT_TRUNK_SIZE);
    }

    public InputStreamToPacketUnidirectionalPipe(InputStream is, Proxy.Metadata metadata, int trunkSize) {
        super();
        this.is = is;
        this.metadata = metadata;
        this.trunkSize = trunkSize;

        this.packetBuilder = Proxy.Packet.newBuilder().setHeader(metadata);
        this.dataBuilder = Proxy.Data.newBuilder();
        this.hasReadBefore = false;
    }

    @Override
    public Proxy.Packet read() {
        LOGGER.info("read for the {} time", ++counter);

        ByteString value = null;
        ByteString cur = null;
        Proxy.Packet packet = null;
        int readCount = 0;
        int curSize = 0;

        try {
            value = ByteString.EMPTY;
            while (readCount++ < MAX_READ_COUNT && !isDrained()) {
                curSize = trunkSize - value.size();
                cur = ByteString.readFrom(ByteStreams.limit(is, curSize));
                value = value.concat(cur);
                if (value.size() >= trunkSize) {
                    break;
                }
            }

            if (hasReadBefore && value.size() == 0) {
                return null;
            }

            packet = packetBuilder
                    .setBody(dataBuilder.setValue(value))
                    .build();

            hasReadBefore = true;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return packet;
    }

    @Override
    public Object read(long timeout, TimeUnit unit) {
        throw new UnsupportedOperationException("Operation not supported");
    }

    @Override
    public void write(Object o) {
        throw new UnsupportedOperationException("Operation not supported");
    }

    @Override
    public boolean isDrained() {
        boolean result = true;
        if (isClosed() || super.isDrained()) {
            return result;
        }

        try {
            for (int i = 0; i < MAX_EMPTY_READ_COUNT; ++i) {
                if (is.available() > 0) {
                    result = false;
                    break;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        if (result == true) {
            this.setDrained();
        }

        return result;
    }

    @Override
    public synchronized void close() {
        super.close();
        IOUtils.closeQuietly(is);
    }
}
