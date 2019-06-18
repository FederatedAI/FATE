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

package com.webank.ai.fate.serving.manger;

import com.webank.ai.fate.core.utils.Configuration;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class InferenceWorkerManager {
    private static ThreadPoolExecutor threadPoolExecutor;
    private static final Logger LOGGER = LogManager.getLogger();

    static {
        LinkedBlockingQueue<Runnable> taskQueue = new LinkedBlockingQueue<>(10);
        threadPoolExecutor = new ThreadPoolExecutor(
                Configuration.getPropertyInt("inferenceWorkerThreadNum"),
                Configuration.getPropertyInt("inferenceWorkerThreadNum"),
                60,
                TimeUnit.SECONDS,
                taskQueue,
                new InferenceWorkerThreadFactory(),
                new InferenceWorkerThreadRejectedPolicy()
        );
    }

    public static void exetute(Runnable task) {
        threadPoolExecutor.execute(task);
    }

    static class InferenceWorkerThreadFactory implements ThreadFactory {
        private final AtomicInteger mThreadNum = new AtomicInteger(1);

        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r, "inference-worker-thread-" + mThreadNum.getAndIncrement());
            LOGGER.info("{} thead has ben created.", t.getName());
            return t;
        }
    }

    public static class InferenceWorkerThreadRejectedPolicy implements RejectedExecutionHandler {
        public void rejectedExecution(Runnable r, ThreadPoolExecutor e) {
            LOGGER.info("{} rejected.", r.toString());
        }
    }

    public static void prestartAllCoreThreads() {
        threadPoolExecutor.prestartAllCoreThreads();
    }
}
