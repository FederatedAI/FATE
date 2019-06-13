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

package com.webank.ai.eggroll.framework;

import com.webank.ai.eggroll.core.factory.DefaultGrpcServerFactory;
import com.webank.ai.eggroll.core.server.BaseEggRollServer;
import com.webank.ai.eggroll.core.server.DefaultServerConf;
import com.webank.ai.eggroll.framework.meta.service.api.grpc.server.ClusterMetaServiceImpl;
import com.webank.ai.eggroll.framework.meta.service.api.grpc.server.StorageMetaServiceImpl;
import io.grpc.Server;
import org.apache.commons.cli.CommandLine;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

public class MetaService extends BaseEggRollServer {
    public static void main(String[] args) throws Exception {
        String confFilePath = null;
        CommandLine cmd = parseArgs(args);

        if (cmd == null) {
            return;
        }

        confFilePath = cmd.getOptionValue("c");

        ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext-meta-service.xml");

        DefaultGrpcServerFactory serverFactory = context.getBean(DefaultGrpcServerFactory.class);
        DefaultServerConf serverConf = (DefaultServerConf) serverFactory.parseConfFile(confFilePath);

        StorageMetaServiceImpl storageMetaService = context.getBean(StorageMetaServiceImpl.class);
        ClusterMetaServiceImpl clusterMetaService = context.getBean(ClusterMetaServiceImpl.class);

        serverConf
                .addService(storageMetaService)
                .addService(clusterMetaService);

        Server server = serverFactory.createServer(serverConf);

        server.start();
        server.awaitTermination();
    }
}
