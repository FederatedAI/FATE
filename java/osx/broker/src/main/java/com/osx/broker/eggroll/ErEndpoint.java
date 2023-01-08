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
package com.osx.broker.eggroll;
import com.webank.eggroll.core.meta.Meta;

public class ErEndpoint extends BaseProto<Meta.Endpoint> {

    String host;
    int port;

    public ErEndpoint(String ip, int port) {
        this.host = ip;
        this.port = port;
    }

    public static ErEndpoint parseFromPb(Meta.Endpoint endpoint) {
        if (endpoint != null) {
            ErEndpoint erEndpoint = new ErEndpoint(endpoint.getHost(), endpoint.getPort());
            return erEndpoint;
        }
        return null;
    }

    public String getHost() {
        return host;
    }

    public void setHost(String host) {
        this.host = host;
    }

    public int getPort() {
        return port;
    }

    public void setPort(int port) {
        this.port = port;
    }

    @Override
    Meta.Endpoint toProto() {
        return Meta.Endpoint.newBuilder().setHost(host).setPort(port).build();

    }
}
