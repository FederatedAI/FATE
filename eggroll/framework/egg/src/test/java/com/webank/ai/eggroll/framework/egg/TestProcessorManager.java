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

package com.webank.ai.eggroll.framework.egg;


import com.webank.ai.eggroll.core.factory.GrpcServerFactory;
import com.webank.ai.eggroll.core.server.ServerConf;
import com.webank.ai.eggroll.core.utils.RuntimeUtils;
import com.webank.ai.eggroll.framework.egg.node.manager.ProcessorManager;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.util.ArrayList;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:applicationContext-egg.xml"})
public class TestProcessorManager {
    @Autowired
    private ProcessorManager processorManager;
    @Autowired
    private ServerConf serverConf;
    @Autowired
    private GrpcServerFactory grpcServerFactory;
    @Autowired
    private RuntimeUtils runtimeUtils;

    @Before
    public void init() throws Exception {
        serverConf = grpcServerFactory.parseConfFile("/Users/max-webank/git/FATE/arch/eggroll/egg/src/main/resources/egg.properties");
    }

    @Test
    public void testGet() throws Exception {
        int port = processorManager.get();

        System.out.println(port);
    }

    @Test
    public void testGetAll() throws Exception {
        ArrayList<Integer> all = processorManager.getAllPossible();

        for (Integer port : all) {
            System.out.println(port);
        }
    }

    @Test
    public void testKill() {
        processorManager.kill(50001);
    }

    @Test
    public void testKillAll() {
        processorManager.killAll();
    }
}
