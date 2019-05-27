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

package utils;

import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.api.networking.proxy.Proxy;

import java.util.Arrays;
import java.util.List;

public class Utils {
    public static final String DELIMITER = "-";
    public static final String DST = "dst";
    public static final String SRC = "src";
    public static final Proxy.HeartbeatResponse DEFAULT_HEARTBEAT_RESPONSE;

    static {
        DEFAULT_HEARTBEAT_RESPONSE = Proxy.HeartbeatResponse.newBuilder().setOperation(Proxy.Operation.RUN).build();
    }

    public static String genEndpointKey(BasicMeta.Endpoint endpoint) {
        List<String> elements
                = Arrays.asList(endpoint.getIp(), endpoint.getHostname(), String.valueOf(endpoint.getPort()));

        return String.join(DELIMITER, elements);
    }

    public static String genStorageKey(String taskId, BasicMeta.Endpoint dst, BasicMeta.Endpoint src) {
        List<String> elements = Arrays.asList(taskId,
                DST, genEndpointKey(dst),
                SRC, genEndpointKey(src));


        return String.join(DELIMITER, elements);
    }

}
