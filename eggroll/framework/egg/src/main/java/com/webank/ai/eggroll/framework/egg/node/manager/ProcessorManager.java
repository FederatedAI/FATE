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

package com.webank.ai.eggroll.framework.egg.node.manager;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.webank.ai.eggroll.core.serdes.impl.GeneralJsonBytesSerDes;
import com.webank.ai.eggroll.core.server.ServerConf;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.core.utils.RandomUtils;
import com.webank.ai.eggroll.core.utils.RuntimeUtils;
import com.webank.ai.eggroll.framework.egg.node.sandbox.ProcessorOperator;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

@Component
/**
 *
 */
public class ProcessorManager {
    @Autowired
    private ProcessorOperator processorOperator;
    @Autowired
    private RandomUtils randomUtils;
    @Autowired
    private GeneralJsonBytesSerDes serDes;
    @Autowired
    private RuntimeUtils runtimeUtils;
    @Autowired
    private ServerConf serverConf;
    @Autowired
    private ErrorUtils errorUtils;

    private Set<Integer> availableProcessors;
    private ArrayList<Integer> availableProcessorMirror;
    private volatile int lastPort;
    private final Object availableProcessorsLock;
    private final Path statusPath;
    private int maxProcessorCount;
    private AtomicInteger lastScheduledProcessor;

    private static final int START_PORT_NOT_INCLUDED = 50000;
    private static final String statusFileLocation = "/tmp/FATE/node-manager/processor-manager";
    private static final int PROCESSOR_COUNT = Runtime.getRuntime().availableProcessors();
    private static final Logger LOGGER = LogManager.getLogger();

    public ProcessorManager() {
        availableProcessors = Sets.newConcurrentHashSet();
        availableProcessorMirror = Lists.newArrayListWithExpectedSize(PROCESSOR_COUNT);
        lastPort = START_PORT_NOT_INCLUDED;
        availableProcessorsLock = new Object();

        lastScheduledProcessor = new AtomicInteger(0);
        statusPath = Paths.get(statusFileLocation);
    }

    @PostConstruct
    public void init() {
        if (Files.exists(statusPath)) {
            LOGGER.info("[EGG][PROCESSOR][MANAGER] restoring processors");
            try {
                byte[] previousStatus = Files.readAllBytes(statusPath);
                if (previousStatus == null || previousStatus.length == 0) {
                    return;
                }

                ArrayList<Integer> deserializedFileContent = serDes.deserialize(previousStatus, ArrayList.class);
                synchronized (availableProcessorsLock) {
                    availableProcessors = Sets.newConcurrentHashSet(deserializedFileContent);

                    checkAvailable();
                }
            } catch (IOException e) {
                Thread.currentThread().interrupt();
            }
        }

        LOGGER.info("[EGG][PROCESSOR][MANAGER] restored processors: {}", availableProcessors);
    }

    /**
     *
     * @return port of available processor
     */
    public int get() {
        initProcessorCount();
        int resultPort = lastPort + 1;
        if (availableProcessors.size() < maxProcessorCount) {
            boolean created = false;
            synchronized (availableProcessorsLock) {
                while (!created && availableProcessors.size() < maxProcessorCount) {
                    try {
                        Process process = processorOperator.start(resultPort);
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

                try {
                    writeStatusFile();
                } catch (IOException e) {
                    Thread.currentThread().interrupt();
                    kill(resultPort);
                }
            }
        } else {
            int target = lastScheduledProcessor.getAndIncrement();
            resultPort = availableProcessorMirror.get(target % maxProcessorCount);
        }

        return resultPort;
    }

    public ArrayList<Integer> getAllPossible() {
        initProcessorCount();

        if (availableProcessors.size() != maxProcessorCount) {
            synchronized (availableProcessorsLock) {
                int beforeAdjustCount = availableProcessors.size();
                if (availableProcessors.size() < maxProcessorCount) {
                    while (availableProcessors.size() < maxProcessorCount && availableProcessors.size() < 128) {
                        get();
                    }
                } else if (availableProcessors.size() > maxProcessorCount) {
                    while (availableProcessors.size() > maxProcessorCount) {
                        kill(availableProcessorMirror.get(availableProcessors.size() - 1));
                    }
                    availableProcessorMirror = Lists.newArrayList(availableProcessorMirror);
                }

                LOGGER.info("[EGG][PROCESSOR][MANAGER] all possible count before adjusted: {}, after adjusted: {}, ports: {}",
                        beforeAdjustCount, availableProcessors.size(), availableProcessors);
            }
        }

        return Lists.newArrayList(availableProcessorMirror);
    }

    public boolean kill(int port) {
        boolean result = false;
        synchronized (availableProcessorsLock) {
            if (availableProcessors.contains(port)) {
                try {
                    result = processorOperator.stop(port);
                    availableProcessors.remove(port);
                } catch (IOException | InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(e);
                }

                try {
                    writeStatusFile();
                } catch (IOException e) {
                    Thread.currentThread().interrupt();
                    LOGGER.error(errorUtils.getStackTrace(e));
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

            try {
                writeStatusFile();
            } catch (IOException e) {
                Thread.currentThread().interrupt();
                LOGGER.error(errorUtils.getStackTrace(e));
            }
        }
        return result;
    }

    private synchronized void writeStatusFile() throws IOException {
        synchronized (availableProcessorsLock) {
            if (Files.notExists(statusPath)) {
                Files.createDirectories(statusPath.getParent());
                Files.createFile(statusPath);
            }
            Files.write(statusPath, serDes.serialize(Lists.newArrayList(availableProcessors)),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        }
    }

    private synchronized void checkAvailable() {
        Set<Integer> refreshedAvailableProcessors = Sets.newConcurrentHashSet();
        synchronized (availableProcessorsLock) {
            for (Integer port : availableProcessors) {
                if (!runtimeUtils.isPortAvailable(port)) {
                    refreshedAvailableProcessors.add(port);
                }
            }

            availableProcessors = refreshedAvailableProcessors;
        }
        if (availableProcessors.size() >= maxProcessorCount) {
            availableProcessorMirror = Lists.newArrayList(availableProcessors);
        }
    }

    private void initProcessorCount() {
        if (maxProcessorCount <= 0) {
            int userDefinedMaxProcessorCount = Integer.valueOf(
                    serverConf.getProperties().getProperty("max.processors.count", "1"));

            maxProcessorCount = Math.min(userDefinedMaxProcessorCount, PROCESSOR_COUNT);
            if (maxProcessorCount < 1) {
                maxProcessorCount = 1;
            }
            LOGGER.info("[EGG][PROCESS][MANAGER] user defined processor count: {}, processor count: {}, final max processor count: {}",
                    userDefinedMaxProcessorCount, PROCESSOR_COUNT, maxProcessorCount);
        }
    }
}
