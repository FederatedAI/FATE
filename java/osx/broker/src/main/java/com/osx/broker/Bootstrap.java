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

package com.osx.broker;


import com.google.common.collect.Lists;
import com.osx.core.constant.Dict;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.StreamLimitMode;
import com.osx.core.jvm.JvmInfoCounter;
import com.osx.core.utils.JsonUtil;
import com.osx.core.utils.NetUtils;
import com.osx.core.utils.ServerUtil;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Properties;


public class Bootstrap {
    static Logger logger = LoggerFactory.getLogger(Bootstrap.class);

    static CommandLine  commandLine;

    public static void main(String[] args) {
        try {
            Options options = ServerUtil.buildCommandlineOptions(new Options());
            commandLine = ServerUtil.parseCmdLine("Transfer", args, buildCommandlineOptions(options),
                    new PosixParser());
            String filePath = commandLine.getOptionValue('c');
            logger.info("try to parse config file {}",filePath);
            if(StringUtils.isEmpty(filePath)){
                System.err.println("config file is not set ,please use -c to set the config file path");
                System.exit(-1);
            }
            parseConfig(filePath);
            Bootstrap bootstrap = new Bootstrap();
            bootstrap.start(args);
            Thread shutDownThread = new Thread(() -> bootstrap.stop());

            shutDownThread.setDefaultUncaughtExceptionHandler(new Thread.UncaughtExceptionHandler() {
                @Override
                public void uncaughtException(Thread t, Throwable e) {
                   logger.error("0000000000000000",e);
                }
            });

            Runtime.getRuntime().addShutdownHook(shutDownThread);
//           Thread.sleep(10000);
//            bootstrap.stop();

        } catch (Exception ex) {
            ex.printStackTrace();
            System.exit(1);
        }
    }

    private static Options buildCommandlineOptions(final Options options) {
        Option opt = new Option("c", "configFile", true, "config properties file");
        opt.setRequired(false);
        options.addOption(opt);
        return options;
    }

    public static void parseConfig(String configFilePath) {
        try {
            File file = new File(configFilePath);
            Properties environment = new Properties();
            try (InputStream inputStream = new BufferedInputStream(new FileInputStream(file))) {
                environment.load(inputStream);
            } catch (FileNotFoundException e) {
                logger.error("profile broker.properties not found");
                throw e;
            } catch (IOException e) {
                logger.error("parse config error, {}", e.getMessage());
                throw e;
            }

            MetaInfo.PROPERTY_ROOT_PATH = new File("").getCanonicalPath();
            MetaInfo.PROPERTY_ROUTE_TABLE = environment.getProperty(Dict.PROPERTY_ROUTE_TABLE);
            MetaInfo.PROPERTY_SERVER_CERTCHAIN_FILE = environment.getProperty(Dict.PROPERTY_SERVER_CERTCHAIN_FILE);
            MetaInfo.PROPERTY_SERVER_PRIVATEKEY_FILE = environment.getProperty(Dict.PROPERTY_SERVER_PRIVATEKEY_FILE);
            MetaInfo.PROPERTY_SERVER_CA_FILE = environment.getProperty(Dict.PROPERTY_SERVER_CA_FILE);
            MetaInfo.PROPERTY_TLS_PORT =  Integer.valueOf(environment.getProperty(Dict.PROPERTY_TLS_PORT,"9883"));
            MetaInfo.PROPERTY_PORT = Integer.valueOf(environment.getProperty(Dict.PORT,"9889"));
            MetaInfo.PROPERTY_PRINT_INPUT_DATA = Boolean.valueOf(environment.getProperty(Dict.PROPERTY_PRINT_INPUT_DATA, "false"));
            MetaInfo.PROPERTY_PRINT_OUTPUT_DATA = Boolean.valueOf(environment.getProperty(Dict.PROPERTY_PRINT_OUTPUT_DATA, "false"));
            MetaInfo.PROPERTY_USER_HOME = System.getProperty("user.home") ;
            MetaInfo.PROPERTY_NEGOTIATIONTYPE = environment.getProperty(Dict.PROPERTY_NEGOTIATIONTYPE,"PLAINTEXT");
            MetaInfo.PROPERTY_TRANSFER_FILE_PATH_PRE = environment.getProperty(Dict.PROPERTY_TRANSFER_FILE_PATH,MetaInfo.PROPERTY_USER_HOME  + "/.fate/transfer_file");
            MetaInfo.PROPERTY_TRANSFER_FILE_CACHE_SIZE = environment.getProperty(Dict.PROPERTY_TRANSFER_FILE_CACHE_SIZE) != null ? Integer.parseInt(environment.getProperty(Dict.PROPERTY_TRANSFER_FILE_CACHE_SIZE)) : 1<<27;
            MetaInfo.PROPERTY_USE_DIRECT_CACHE = Boolean.parseBoolean(environment.getProperty(Dict.PROPERTY_USE_DIRECT_CACHE, "false"));
            MetaInfo.PROPERTY_MAX_TRANSFER_CACHE_SIZE = environment.getProperty(Dict.PROPERTY_MAX_TRANSFER_CACHE_SIZE) != null ? Integer.parseInt(environment.getProperty(Dict.PROPERTY_MAX_TRANSFER_CACHE_SIZE)) : 1<<30;
            MetaInfo.PROPERTY_GRPC_ONCOMPLETED_WAIT_TIMEOUT = Integer.parseInt(environment.getProperty(Dict.PROPERTY_GRPC_ONCOMPLETED_WAIT_TIMEOUT,"600"));
            MetaInfo.PROPERTY_USE_QUEUE_MODEL = Boolean.valueOf(environment.getProperty(Dict.PROPERTY_USE_QUEUE_MODEL, "false"));
            MetaInfo.PROPERTY_STREAM_LIMIT_MODE = environment.getProperty(Dict.PROPERTY_STREAM_LIMIT_MODE, StreamLimitMode.LOCAL.name());
            MetaInfo.PROPERTY_STREAM_LIMIT_MAX_TRY_TIME = Integer.parseInt(environment.getProperty(Dict.PROPERTY_STREAM_LIMIT_MAX_TRY_TIME,"10"));
            MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONCURRENT_CALL_PER_CONNECTION = Integer.parseInt(environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_MAX_CONCURRENT_CALL_PER_CONNECTION,"1000"));
            MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_INBOUND_MESSAGE_SIZE = environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_MAX_INBOUND_MESSAGE_SIZE) != null ? Integer.parseInt(environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_MAX_INBOUND_MESSAGE_SIZE)) : (2<<30)-1;
             MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_INBOUND_METADATA_SIZE = environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_MAX_INBOUND_METADATA_SIZE) != null ? Integer.parseInt(environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_MAX_INBOUND_METADATA_SIZE)) : 128<<20;
             MetaInfo.PROPERTY_GRPC_CHANNEL_FLOW_CONTROL_WINDOW = environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_FLOW_CONTROL_WINDOW) != null ? Integer.parseInt(environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_FLOW_CONTROL_WINDOW)) : 128<<20;
            MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIME_SEC = Integer.parseInt(environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIME_SEC,"7200"));
            MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIMEOUT_SEC = Integer.parseInt(environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIMEOUT_SEC,"3600"));
            MetaInfo.PROPERTY_GRPC_CHANNEL_PERMIT_KEEPALIVE_TIME_SEC = Integer.parseInt(environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_PERMIT_KEEPALIVE_TIME_SEC,"120"));
            MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_WITHOUT_CALLS_ENABLED = Boolean.parseBoolean(environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_KEEPALIVE_WITHOUT_CALLS_ENABLED, "false"));
            MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_IDLE_SEC = Integer.parseInt(environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_IDLE_SEC,"86400"));
            MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_SEC = Integer.parseInt(environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_SEC,"86400"));
            MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_GRACE_SEC = Integer.parseInt(environment.getProperty(Dict.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_GRACE_SEC,"86400"));
            MetaInfo.TRANSFER_FATECLOUD_AHTHENTICATION_ENABLED = Boolean.valueOf(environment.getProperty(Dict.TRANSFER_FATECLOUD_AHTHENTICATION_ENABLED,"false"));
            MetaInfo.TRANSFER_FATECLOUD_AUTHENTICATION_USE_CONFIG = Boolean.valueOf(environment.getProperty(Dict.TRANSFER_FATECLOUD_AUTHENTICATION_USE_CONFIG,"false"));
            MetaInfo.TRANSFER_FATECLOUD_AUTHENTICATION_ROLE = environment.getProperty(Dict.TRANSFER_FATECLOUD_AUTHENTICATION_ROLE,"guest");
            MetaInfo.TRANSFER_FATECLOUD_AUTHENTICATION_URI = environment.getProperty(Dict.TRANSFER_FATECLOUD_AUTHENTICATION_URI, "/cloud-manager/api/site/rollsite/checkPartyId");
            MetaInfo.TRANSFER_FATECLOUD_AUTHENTICATION_APPKEY = environment.getProperty(Dict.TRANSFER_FATECLOUD_AUTHENTICATION_APPKEY, "");
            MetaInfo.TRANSFER_FATECLOUD_AUTHENTICATION_APPSERCRET = environment.getProperty(Dict.TRANSFER_FATECLOUD_AUTHENTICATION_APPSERCRET, "");
            MetaInfo.TRANSFER_FATECLOUD_SECRET_INFO_URL = environment.getProperty(Dict.TRANSFER_FATECLOUD_SECRET_INFO_URL,"http://localhost:9091/fate-manager/api/site/secretinfo");
            MetaInfo.TRANSFER_FATECLOUD_AUTHENTICATION_URL = environment.getProperty(Dict.TRANSFER_FATECLOUD_AUTHENTICATION_URL,"http://localhost:8999/cloud-manager/api/site/rollsite/checkPartyId");
            MetaInfo.PROPERTY_SELF_PARTY.addAll(Lists.newArrayList(environment.getProperty(Dict.PROPERTY_SELF_PARTY,"").split(",")));;
            MetaInfo.PRPPERTY_QUEUE_MAX_FREE_TIME = Integer.parseInt(environment.getProperty(Dict.PRPPERTY_QUEUE_MAX_FREE_TIME,"60000000"));
            MetaInfo.INSTANCE_ID = NetUtils.getLocalHost()+":"+MetaInfo.PROPERTY_PORT;
            MetaInfo.PROPERTY_DEPLOY_MODE=environment.getProperty(Dict.PROPERTY_DEPLOY_MODE);
            MetaInfo.PROPERTY_CLUSTER_MANAGER_ADDRESS = environment.getProperty(Dict.PROPERTY_CLUSTER_MANAGER_ADDRESS);
            MetaInfo.PROPERTY_DLEDGER_PEER = environment.getProperty(Dict.PROPERTY_DLEDGER_PEER);
            MetaInfo.PROPERTY_DLEDGER_SELF = environment.getProperty(Dict.PROPERTY_DLEDGER_SELF);
            MetaInfo.PROPERTY_EGGROLL_CLUSTER_MANANGER_IP = environment.getProperty(Dict.PROPERTY_EGGROLL_CLUSTER_MANANGER_IP);
            MetaInfo.PROPERTY_EGGROLL_CLUSTER_MANANGER_PORT = Integer.parseInt(environment.getProperty(Dict.PROPERTY_EGGROLL_CLUSTER_MANANGER_PORT));
            MetaInfo.PROPERTY_ZK_URL = environment.getProperty(Dict.PROPERTY_ZK_URL);
        } catch (Exception e) {
            logger.error("init MetaInfo error", e);
            System.exit(1);
        }
        logger.info("Meta Info {}", JsonUtil.formatJson(JsonUtil.object2Json(MetaInfo.toMap())));
    }

    public void start(String[] args) {
        ServiceContainer.init();
        JvmInfoCounter.start();



    }

    public void stop() {
        logger.info("try to shutdown server ...");
        if(ServiceContainer.transferQueueManager!=null){
            ServiceContainer.transferQueueManager.destroyAll();
        }
    }

}