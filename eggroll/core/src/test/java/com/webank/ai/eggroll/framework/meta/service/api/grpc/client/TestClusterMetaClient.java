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

package com.webank.ai.eggroll.framework.meta.service.api.grpc.client.client;

import com.webank.ai.eggroll.core.api.grpc.client.crud.ClusterMetaClient;
import com.webank.ai.eggroll.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.eggroll.core.model.NodeStatus;
import com.webank.ai.eggroll.core.model.NodeType;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:applicationContext-core.xml"})
public class TestClusterMetaClient {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private StorageMetaClient storageMetaClient;
    @Autowired
    private ClusterMetaClient clusterMetaClient;

    @Before
    public void testAbstractInit() {
        storageMetaClient.init("127.0.0.1", 8590);
        clusterMetaClient.init("127.0.0.1", 8590);
        LOGGER.info("inited");
    }

    @Test
    public void testGetNodeByNodeId() {
        Node result = storageMetaClient.getNodeByNodeId(1L);
        LOGGER.info(result);
    }

    @Test
    public void testGetFragmentByFragmentId() {
        Fragment result = storageMetaClient.getFragmentByFragmentId(1L);
        LOGGER.info(result);
    }

    @Test
    public void testGetNodeByFragmentId() {
        Node result = storageMetaClient.getNodeByFragmentId(1L);
        LOGGER.info(result);
    }

    @Test
    public void testRegisterNode() {
        Node request = new Node();
        request.setIp("127.0.0.1");
        request.setPort(8250);
        request.setType(NodeType.EGG.name());
        request.setStatus(NodeStatus.HEALTHY.name());

        Node result = clusterMetaClient.registerNode(request);

        LOGGER.info(result);
    }
}
