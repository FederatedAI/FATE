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

package com.webank.ai.fate.networking;

import com.webank.ai.fate.networking.proxy.factory.GrpcServerFactory;
import com.webank.ai.fate.networking.proxy.factory.LocalBeanFactory;
import com.webank.ai.fate.networking.proxy.manager.ServerConfManager;
import com.webank.ai.fate.networking.proxy.model.ServerConf;
import io.grpc.Server;
import org.apache.commons.cli.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

public class Proxy {
    private static final Logger LOGGER = LogManager.getLogger();

    public static void main(String[] args) throws Exception {
        Options options = new Options();
        Option config = Option.builder("c")
                .argName("file")
                .longOpt("config")
                .hasArg()
                .numberOfArgs(1)
                .required()
                .desc("configuration file")
                .build();

        options.addOption(config);

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

        String confFilePath = cmd.getOptionValue("c");

        ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext-proxy.xml");

        LocalBeanFactory localBeanFactory = context.getBean(LocalBeanFactory.class);
        localBeanFactory.setApplicationContext(context);
        GrpcServerFactory serverFactory = context.getBean(GrpcServerFactory.class);

        Server server = serverFactory.createServer(confFilePath);

        ServerConfManager serverConfManager = context.getBean(ServerConfManager.class);
        ServerConf serverConf = serverConfManager.getServerConf();

        LOGGER.info("Server started listening on port: {}", serverConf.getPort());
        LOGGER.info("server conf: {}", serverConf);

        server.start();
        server.awaitTermination();
    }
}
