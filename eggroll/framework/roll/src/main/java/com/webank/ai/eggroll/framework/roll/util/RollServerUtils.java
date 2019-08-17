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

package com.webank.ai.eggroll.framework.roll.util;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.eggroll.core.constant.RuntimeConstants;
import com.webank.ai.eggroll.core.model.NodeStatus;
import com.webank.ai.eggroll.core.model.NodeType;
import com.webank.ai.eggroll.core.server.ServerConf;
import com.webank.ai.eggroll.core.utils.TypeConversionUtils;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Properties;

@Component
public class RollServerUtils {
    @Autowired
    private ServerConf serverConf;
    @Autowired
    private StorageMetaClient storageMetaClient;
    @Autowired
    private TypeConversionUtils typeConversionUtils;

    private BasicMeta.Endpoint metaServiceEndpoint;

    private BasicMeta.Endpoint rollEndpoint;

    private static final Logger LOGGER = LogManager.getLogger();

    public void init() {
        storageMetaClient.init(getMetaServiceEndpoint());
    }

    public synchronized BasicMeta.Endpoint getMetaServiceEndpoint() {
        if (metaServiceEndpoint == null) {
            try {
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
                    throw new IllegalArgumentException("meta.service.port cannot be null");
                }

                metaServiceEndpoint = builder.build();
            } catch (Exception e) {
                LOGGER.info("[ROLL][INIT] init meta-service endpoint failed. using default");
                metaServiceEndpoint = RuntimeConstants.getLocalEndpoint(8590);
            }
        }

        return metaServiceEndpoint;
    }

    public synchronized BasicMeta.Endpoint getRollEndpoint() {
        if (rollEndpoint == null) {
            try {
                init();
                Node node = new Node();
                node.setType(NodeType.ROLL.name());
                node.setStatus(NodeStatus.HEALTHY.name());
                List<Node> nodes = storageMetaClient.getNodes(node);

                if (nodes == null || nodes.isEmpty()) {
                    throw new IllegalStateException("no valid roll node exists");
                }


                rollEndpoint = typeConversionUtils.toEndpoint(nodes.get(0));
            } catch (Exception e) {
                LOGGER.info("[ROLL][INIT] init roll endpoint failed. using default");
                rollEndpoint = RuntimeConstants.getLocalEndpoint(8011);
            }
        }

        return rollEndpoint;
    }
}
