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
package org.fedai.osx.broker;

import com.google.inject.Guice;
import com.google.inject.Injector;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.broker.ptp.PtpUnaryCallService;
import org.fedai.osx.broker.queue.TransferQueueManager;
import org.fedai.osx.broker.server.OsxServer;
import org.fedai.osx.broker.util.ApplicationStartedRunnerUtils;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.jvm.JvmInfoCounter;
import org.fedai.osx.core.utils.PropertiesUtil;
import org.fedai.osx.core.utils.ServerUtil;
import org.fedai.osx.guice.BrokerModule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

public class Bootstrap {
    static Logger logger = LoggerFactory.getLogger(Bootstrap.class);
    static CommandLine commandLine;
    static Object lockObject= new Object();
    static Injector injector;
    public static void main(String[] args) {
        try {
             injector = Guice.createInjector(new BrokerModule() );
            System.err.println(injector.getAllBindings());
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
            List<String> packages = new ArrayList<>();
            packages.add(Bootstrap.class.getPackage().getName());
            ApplicationStartedRunnerUtils.run(injector, packages, args);

            injector.getInstance(OsxServer.class).start();
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
//        ServiceContainer.init();
        JvmInfoCounter.start();
    }

    public void stop() {
        logger.info("try to shutdown server ...");
        if (injector != null) {
            TransferQueueManager  transferQueueManager = injector.getInstance(TransferQueueManager.class);
            if(transferQueueManager!=null)
                    transferQueueManager.destroyAll();
        }
    }

}