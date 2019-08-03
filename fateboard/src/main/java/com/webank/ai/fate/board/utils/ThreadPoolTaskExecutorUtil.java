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
package com.webank.ai.fate.board.utils;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.core.task.TaskRejectedException;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.util.concurrent.ListenableFuture;

import java.util.concurrent.Callable;


public class ThreadPoolTaskExecutorUtil {

    private static Logger LOGGER = LogManager.getLogger();

    public static ListenableFuture<?> submitListenable(ThreadPoolTaskExecutor executor, Callable callable, int[] sleepTimes, int[] tryCount) {

        ListenableFuture<?> resultListenableFuture = null;

        try {
            resultListenableFuture = executor.submitListenable(callable);
        } catch (TaskRejectedException e) {
            boolean success = false;
            for (int i = 0; i < sleepTimes.length && i < tryCount.length; i++) {
                int tryNum = tryCount[i];
                int sleepTime = sleepTimes[i];
                int count = 0;
                do {
                    count++;
                    try {
                        try {
                            Thread.sleep(sleepTime);
                        } catch (InterruptedException e1) {
                            e1.printStackTrace();
                        }
                        resultListenableFuture = executor.submitListenable(callable);
                        return resultListenableFuture;
                    } catch (TaskRejectedException taskException) {
                        LOGGER.error("[FEDERATION][THREADPOOL] submit exception,sleep {} tryCount {}", sleepTime, count);
                    }
                } while (count < tryNum);
            }
            if (resultListenableFuture == null) {
                throw e;
            }
        }
        return resultListenableFuture;
    }


    public static ListenableFuture<?> submitListenable(ThreadPoolTaskExecutor executor, Runnable callable, int[] sleepTimes, int[] tryCount) {

        ListenableFuture<?> resultListenableFuture = null;

        try {
            resultListenableFuture = executor.submitListenable(callable);
        } catch (TaskRejectedException e) {
            for (int i = 0; i < sleepTimes.length && i < tryCount.length; i++) {
                int tryNum = tryCount[i];
                int sleepTime = sleepTimes[i];
                int count = 0;
                do {
                    count++;
                    try {
                        try {
                            Thread.sleep(sleepTime);
                        } catch (InterruptedException e1) {
                            e1.printStackTrace();
                        }
                        resultListenableFuture = executor.submitListenable(callable);
                        return resultListenableFuture;
                    } catch (TaskRejectedException taskException) {
                        LOGGER.error("[FEDERATION][THREADPOOL] submit exception,sleep {} tryCount {}", sleepTime, count);
                    }
                } while (count < tryNum);
            }
            if (resultListenableFuture == null) {
                throw e;
            }
        }
        return resultListenableFuture;
    }
}
