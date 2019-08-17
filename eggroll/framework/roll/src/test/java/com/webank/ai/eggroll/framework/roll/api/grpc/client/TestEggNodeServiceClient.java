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

package com.webank.ai.eggroll.framework.roll.api.grpc.client;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath*:applicationContext-roll.xml"})
public class TestEggNodeServiceClient {
    @Autowired
    private EggNodeManagerClient eggNodeManagerClient;

    private Node node;

    public TestEggNodeServiceClient() {
        node = new Node();
        node.setIp("127.0.0.1");
        node.setPort(7888);
    }

    @Test
    public void testGetProcessor() {
        BasicMeta.Endpoint result = eggNodeManagerClient.getProcessor(node);
        System.out.println(result);
    }
}
