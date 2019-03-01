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

package com.webank.ai.fate.eggroll.egg.node.manager;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.webank.ai.fate.core.utils.RandomUtils;
import com.webank.ai.fate.eggroll.egg.node.sandbox.ProcessorOperator;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Set;

@Component
/**
 *
 */
public class ProcessorManager {
    @Autowired
    private ProcessorOperator processorOperator;
    @Autowired
    private RandomUtils randomUtils;

    private Set<Integer> availableProcessors;
    private ArrayList<Integer> availableProcessorMirror;
    private volatile int lastPort;
    private final Object availableProcessorsLock;

    private static final int maxProcessorCount =
            Runtime.getRuntime().availableProcessors() > 1 ? Runtime.getRuntime().availableProcessors() - 1 : 1;

    public ProcessorManager() {
        availableProcessors = Sets.newConcurrentHashSet();
        availableProcessorMirror = Lists.newArrayListWithExpectedSize(maxProcessorCount);
        lastPort = 50000;
        availableProcessorsLock = new Object();
    }

    /**
     *
     * @return port of available processor
     */
    public int get() {
        int resultPort = lastPort + 1;
        if (availableProcessors.size() < maxProcessorCount) {
            boolean created = false;
            synchronized (availableProcessorsLock) {
                while (!created && availableProcessors.size() < maxProcessorCount) {
                    try {
                        Process process = processorOperator.startProcessor(resultPort);
                        if (process != null) {
                            created = true;
                        }
                    } catch (IOException ignore) {
                        ++resultPort;
                    }
                }

                lastPort = resultPort;
                availableProcessors.add(resultPort);
                if (availableProcessors.size() >= maxProcessorCount) {
                    availableProcessorMirror = Lists.newArrayList(availableProcessors);
                }
            }
        } else {
            int target = randomUtils.nextInt(0, maxProcessorCount - 1);
            resultPort = availableProcessorMirror.get(target);
        }

        return resultPort;
    }

    public ArrayList<Integer> getAllAvailable() {
        while (availableProcessorMirror.size() == 0 && availableProcessors.size() < 10) {
            get();
        }
        return Lists.newArrayList(availableProcessorMirror);
    }

    public boolean kill(int port) {
        boolean result = false;
        synchronized (availableProcessorsLock) {
            if (availableProcessors.contains(port)) {
                try {
                    result = processorOperator.stopProcessor(port);
                    availableProcessors.remove(port);
                } catch (IOException | InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(e);
                }
            } else {
                result = false;
            }
        }

        return result;
    }

    public boolean killAll() {
        boolean result = true;
        synchronized (availableProcessorsLock) {
            for (Integer port : availableProcessors) {
                result = kill(port);
            }
        }
        return result;
    }
}
