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

package com.webank.ai.fate.driver.federation.utils;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.core.server.ServerConf;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.Properties;

@Component
public class FederationServerUtils {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private ServerConf serverConf;
    private BasicMeta.Endpoint metaServiceEndpoint;

    public BasicMeta.Endpoint getMetaServiceEndpoint() {
        if (metaServiceEndpoint == null) {
            BasicMeta.Endpoint.Builder builder = BasicMeta.Endpoint.newBuilder();

            Properties properties = serverConf.getProperties();

            String metaServiceIp = properties.getProperty("meta.service.ip", null);
            String metaServiceHost = properties.getProperty("meta.service.host", null);

            if (StringUtils.isAllBlank(metaServiceIp, metaServiceHost)) {
                throw new IllegalArgumentException("meta.service.ip and meta.service.host cannot be all null");
            }

            if (StringUtils.isNotBlank(metaServiceIp)) {
                builder.setIp(metaServiceIp);
            }

            if (StringUtils.isNotBlank(metaServiceHost)) {
                builder.setHostname(metaServiceHost);
            }

            String metaServicePort = properties.getProperty("meta.service.port", null);
            if (StringUtils.isNotBlank(metaServicePort)) {
                builder.setPort(Integer.valueOf(metaServicePort));
            } else {
                throw new IllegalArgumentException("meta.service.ip cannot be null");
            }

            metaServiceEndpoint = builder.build();
        }

        return metaServiceEndpoint;
    }
}
