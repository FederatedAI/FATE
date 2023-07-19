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

package com.osx.core.service;


import io.grpc.ManagedChannel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

public class InboundPackage<T> {

    static Logger logger = LoggerFactory.getLogger(InboundPackage.class);
    ManagedChannel managedChannel;


    String source;
    Map head;

    T body;

    public ManagedChannel getManagedChannel() {
        return managedChannel;
    }

    public void setManagedChannel(ManagedChannel managedChannel) {
        this.managedChannel = managedChannel;
    }

    public String getSource() {
        return source;
    }

    public void setSource(String source) {
        this.source = source;
    }

    public Map getHead() {
        return head;
    }

    public void setHead(Map head) {
        this.head = head;
    }

    public T getBody() {
        return body;
    }

    public void setBody(T body) {
        this.body = body;
    }
}
