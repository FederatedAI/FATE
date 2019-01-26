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

package com.webank.ai.fate.core.factory;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.springframework.stereotype.Component;

@Component
public class ServerUtilitiesFactory {
    public Options createDefaultOptions() {
        Options result = new Options();
        Option config = Option.builder("c")
                .argName("file")
                .longOpt("config")
                .hasArg()
                .numberOfArgs(1)
                .required()
                .desc("configuration file")
                .build();

        Option help = Option.builder("h")
                .longOpt("help")
                .desc("print this message")
                .build();

        result.addOption(config)
                .addOption(help);

        return result;
    }
}
