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

package com.webank.ai.fate.networking.proxy.manager;


import com.webank.ai.fate.networking.proxy.util.ToStringUtils;
import org.apache.commons.text.StringSubstitutor;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.scheduling.concurrent.ThreadPoolTaskScheduler;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

@Component
public class ExecutorManager {
    private static final String LOG_TEMPLATE;
    private static final Logger LOGGER = LogManager.getLogger("stat");

    static {
        LOG_TEMPLATE = "executor stat: pool name: ${name}, " +
                "size: ${size}, " +
                "active count: ${activeCount}";
    }

    @Autowired
    private ThreadPoolTaskExecutor asyncThreadPool;
    @Autowired
    private ThreadPoolTaskExecutor grpcServiceExecutor;
    @Autowired
    private ThreadPoolTaskExecutor grpcClientExecutor;
    @Autowired
    private ThreadPoolTaskScheduler routineScheduler;
    @Autowired
    private ToStringUtils toStringUtils;
    private List<ThreadPoolTaskExecutor> threadPoolExecutors;

    public ExecutorManager() {
        threadPoolExecutors = new LinkedList<>();
    }

    @PostConstruct
    private void init() {
        threadPoolExecutors.add(asyncThreadPool);
        threadPoolExecutors.add(grpcServiceExecutor);
        threadPoolExecutors.add(grpcClientExecutor);
    }

    public void statExecutor() {
        LOGGER.info("------------ executor stat ------------");

        Map<String, String> valuesMap = new HashMap<>(10);
        StringSubstitutor stringSubstitutor = new StringSubstitutor(valuesMap);

        for (ThreadPoolTaskExecutor executor : threadPoolExecutors) {
            valuesMap.clear();
            valuesMap.put("name", executor.getThreadNamePrefix().replace("-", ""));
            valuesMap.put("size", String.valueOf(executor.getPoolSize()));
            valuesMap.put("activeCount", String.valueOf(executor.getActiveCount()));

            String log = stringSubstitutor.replace(LOG_TEMPLATE);
            LOGGER.info(log);
        }
    }
}
