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
import com.osx.core.config.MetaInfo;
import com.osx.core.jvm.JvmInfoCounter;
import com.osx.core.utils.PropertiesUtil;
import com.osx.core.utils.ServerUtil;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.Properties;
public class Bootstrap {
    static Logger logger = LoggerFactory.getLogger(Bootstrap.class);
    static CommandLine commandLine;
    static Object lockObject= new Object();
    public static void main(String[] args) {
        try {
            Options options = ServerUtil.buildCommandlineOptions(new Options());
            commandLine = ServerUtil.parseCmdLine("osx", args, buildCommandlineOptions(options),
                    new PosixParser());
            String configDir = commandLine.getOptionValue('c');
            logger.info("try to parse config dir {}", configDir);
            if (StringUtils.isEmpty(configDir)) {
                System.err.println("config file is not set ,please use -c to set the config file dir path");
                System.exit(-1);
            }
            parseConfig(configDir);
            Bootstrap bootstrap = new Bootstrap();
            bootstrap.start(args);
            Thread shutDownThread = new Thread(bootstrap::stop);
            Runtime.getRuntime().addShutdownHook(shutDownThread);
            synchronized (lockObject){
                lockObject.wait();
            }

        } catch (Exception ex) {
            logger.error("broker start failed ",ex);
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

    public static void parseConfig(String configDir) {
        try {
            MetaInfo.PROPERTY_CONFIG_DIR = configDir;
            String configFilePath =  configDir+ "/broker/broker.properties";
            Properties environment = PropertiesUtil.getProperties(configFilePath);
            MetaInfo.init(environment);
        } catch (Exception e) {
            logger.error("init MetaInfo error", e);
            System.exit(1);
        }
    }

    public void start(String[] args) {
        ServiceContainer.init();
        JvmInfoCounter.start();
    }

    public void stop() {
        logger.info("try to shutdown server ...");
        if (ServiceContainer.transferQueueManager != null) {
            ServiceContainer.transferQueueManager.destroyAll();
        }
    }

}