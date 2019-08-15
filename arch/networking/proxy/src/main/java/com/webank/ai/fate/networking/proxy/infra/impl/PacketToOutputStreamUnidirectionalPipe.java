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

import com.webank.ai.fate.api.networking.proxy.Proxy;
import org.apache.commons.io.IOUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.OutputStream;
import java.util.concurrent.TimeUnit;

@Component
@Scope("prototype")
public class PacketToOutputStreamUnidirectionalPipe extends BasePipe {
    private static final Logger LOGGER = LogManager.getLogger(PacketToOutputStreamUnidirectionalPipe.class);
    private OutputStream os;
    private int counter = 0;

    public PacketToOutputStreamUnidirectionalPipe(OutputStream os) {
        super();
        this.os = os;
    }

    @Override
    public Object read() {
        throw new UnsupportedOperationException("Operation not supported");
    }

    @Override
    public Object read(long timeout, TimeUnit unit) {
        throw new UnsupportedOperationException("Operation not supported");
    }

    @Override
    public void write(Object o) {
        LOGGER.info("write for the {} time", ++counter);
        if (o instanceof Proxy.Packet) {
            Proxy.Packet packet = (Proxy.Packet) o;
            try {
                packet.getBody().getValue().writeTo(os);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        } else {
            throw new IllegalArgumentException("object o is of type: " + o.getClass().getCanonicalName()
                    + ", which is not of type " + Proxy.Packet.class.getCanonicalName());
        }
    }

    @Override
    public boolean isDrained() {
        throw new UnsupportedOperationException("Operation not supported");
    }

    @Override
    public void close() {
        try {
            os.flush();
        } catch (IOException ignore) {
            ;
        }
        IOUtils.closeQuietly(os);
        super.close();
    }
}
