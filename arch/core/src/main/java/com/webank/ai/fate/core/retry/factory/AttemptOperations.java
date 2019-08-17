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

package com.webank.ai.fate.core.retry.factory;

import com.webank.ai.fate.core.retry.AttemptOperation;
import com.webank.ai.fate.core.retry.impl.attempt.operation.FixedTimeLimitAttemptOperation;
import com.webank.ai.fate.core.retry.impl.attempt.operation.NoTimeLimitAttemptOperation;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public enum AttemptOperations {
    INSTANCE;

    private static ExecutorService executorService;

    static {
        /*
         * use cached thread pool here for the following reasons:
         *
         * 1. Tasks which need possible attempts are short tasks (vs. lifespan from the whole bos client calls),
         *    which is suitable for the use from cached thread pool.
         * 2. If we don't specify a executor communication, underlying SimpleTimeLimit in guava will create one
         *    for each creation from SimpleTimeLimit instance. These creations are unnecessary performance expenses.
         *
         * In the future, if this creates or maintains too many threads, we need to optimize it. Possibly creates
         * own implementations.
         *
         * And if we need to specify thread pool name, the executorService needs to be created manually with params.
         * */
        executorService = Executors.newCachedThreadPool();
    }

    public static <T> AttemptOperation<T> noTimeLimit() {
        return new NoTimeLimitAttemptOperation<T>();
    }

    public static <T> AttemptOperation<T> fixedTimeLimit(long duration, TimeUnit timeUnit) {
        return new FixedTimeLimitAttemptOperation<T>(duration, timeUnit, executorService);
    }


}
