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

package com.webank.ai.fate.core.server;


import com.webank.ai.fate.core.factory.ServerUtilitiesFactory;
import org.apache.commons.cli.*;

public abstract class BaseFateServer {
    public static CommandLine parseArgs(String[] args) {
        HelpFormatter formatter = new HelpFormatter();

        ServerUtilitiesFactory serverUtilitiesFactory = new ServerUtilitiesFactory();
        Options options = serverUtilitiesFactory.createDefaultOptions();
        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = null;
        try {
            cmd = parser.parse(options, args);
            if (cmd.hasOption("h")) {
                formatter.printHelp("", options, true);
                return null;
            }
        } catch (ParseException e) {
            formatter.printHelp("", options, true);
        }

        return cmd;
    }
}
