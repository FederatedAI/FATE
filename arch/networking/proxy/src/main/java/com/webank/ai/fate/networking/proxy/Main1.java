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

package com.webank.ai.fate.networking.proxy;

import com.webank.ai.fate.networking.proxy.factory.GrpcServerFactory;
import com.webank.ai.fate.networking.proxy.factory.LocalBeanFactory;
import com.webank.ai.fate.networking.proxy.model.ServerConf;
import io.grpc.Server;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

public class Main1 {
    private static final Logger LOGGER = LogManager.getLogger(Main1.class);

    public static void main(String[] args) throws Exception {
        ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext-proxy.xml");
        LocalBeanFactory localBeanFactory = context.getBean(LocalBeanFactory.class);
        localBeanFactory.setApplicationContext(context);
        GrpcServerFactory serverFactory = context.getBean(GrpcServerFactory.class);

        int port = 8888;

        ServerConf serverConf = context.getBean(ServerConf.class);

        serverConf.setCoordinator("10000");
        serverConf.setSecureServer(false);
        serverConf.setSecureClient(true);
/*

        serverConf.setServerKeyPath("/Users/max-webank/Documents/zmodem/server.key");
        serverConf.setServerCrtPath("/Users/max-webank/Documents/zmodem/server.crt");
*/

        serverConf.setRouteTablePath("src/main/resources/route_tables/route_table1.json");
        serverConf.setPort(port);

        LOGGER.info("Server started listening on port: {}", port);

        Server server = serverFactory.createServer(serverConf);

        LOGGER.info("server conf: {}", serverConf);

        server.start();
        server.awaitTermination();
    }
}
