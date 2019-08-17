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

package com.webank.ai.fate.networking.proxy.util;

import org.apache.commons.cli.Options;
import org.springframework.stereotype.Component;

@Component
public class CommandLineOptions {
    public static final String HELP = "h";
    public static final String PORT = "p";
    public static final String ROUTE_TABLE = "t";
    public static final String SERVER_CRT = "c";
    public static final String SERVER_KEY = "k";
    public static final String ROOT_CRT = "r";
    private final Options options;

    public CommandLineOptions() {
        options = new Options();
        options.addOption("h", "help", false, "Print this usage information");
        options.addOption("p", "port", true, "Port to listen");
        options.addOption("t", "route-table", true, "File to config route rules. Imply using TLS.");
        options.addOption("c", "server-crt", true, "Server certification file. Imply using TLS.");
        options.addOption("k", "server-key", true, "Server private key file. Imply using TLS.");
        options.addOption("r", "root-crt", true, "Root certification file");
    }

    public Options getOptions() {
        return options;
    }
}
