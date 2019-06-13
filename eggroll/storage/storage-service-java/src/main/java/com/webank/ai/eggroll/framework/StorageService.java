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

import com.webank.ai.eggroll.core.server.BaseEggRollServer;
import com.webank.ai.eggroll.framework.storage.service.manager.LMDBStoreManager;
import com.webank.ai.eggroll.framework.storage.service.server.LMDBServicer;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.ServerInterceptors;
import org.apache.commons.cli.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class StorageService extends BaseEggRollServer {
    private static final Logger LOGGER = LogManager.getLogger();

    public static void main(String[] args) throws Exception {
        Options options = new Options();
        Option serverPortOption = Option.builder("p")
                .longOpt("server-port")
                .argName("port")
                .numberOfArgs(1)
                .required()
                .desc("port for the service")
                .build();

        Option dataDirOption = Option.builder("d")
                .longOpt("data-dir")
                .argName("path")
                .numberOfArgs(1)
                .desc("directory to store data")
                .build();

        Option helpOption = Option.builder("h")
                .longOpt("help")
                .desc("print this message")
                .build();

        options.addOption(serverPortOption)
                .addOption(dataDirOption)
                .addOption(helpOption);

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

        int serverPort = Integer.valueOf(cmd.getOptionValue("p"));
        String dataDir = cmd.getOptionValue("d");


        LMDBServicer objectStoreServicer = new LMDBServicer(new LMDBStoreManager(dataDir));
        Server server = ServerBuilder.forPort(serverPort)
                .addService(ServerInterceptors.intercept(objectStoreServicer, new LMDBServicer.KvStoreInterceptor()))
                .maxInboundMessageSize(256 << 20).build();


        LOGGER.info("Server started listening on port: {}, data dir: {}", serverPort, dataDir);
        server.start();
        server.awaitTermination();
    }
}
