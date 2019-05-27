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

package com.webank.ai.fate.core.retry.impl.attempt.operation;

import com.google.common.util.concurrent.SimpleTimeLimiter;
import com.google.common.util.concurrent.TimeLimiter;
import com.webank.ai.fate.core.retry.AttemptOperation;

import javax.annotation.concurrent.Immutable;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

@Immutable
public final class FixedTimeLimitAttemptOperation<T> implements AttemptOperation<T> {
    private final TimeLimiter timeLimiter;
    private final long duration;
    private final TimeUnit timeUnit;

    public FixedTimeLimitAttemptOperation(long duration) {
        this(null, duration, TimeUnit.MILLISECONDS);
    }

    public FixedTimeLimitAttemptOperation(long duration, TimeUnit timeUnit) {
        this(null, duration, timeUnit);
    }

    public FixedTimeLimitAttemptOperation(long duration, TimeUnit timeUnit, ExecutorService executorService) {
        this(SimpleTimeLimiter.create(executorService), duration, timeUnit);
    }

    public FixedTimeLimitAttemptOperation(TimeLimiter timeLimiter, long duration, TimeUnit timeUnit) {
        if (timeLimiter == null) {
            timeLimiter = SimpleTimeLimiter.create(Executors.newSingleThreadExecutor());
        }

        if (timeUnit == null) {
            timeUnit = TimeUnit.MILLISECONDS;
        }

        this.timeLimiter = timeLimiter;
        this.duration = duration;
        this.timeUnit = timeUnit;
    }

    @Override
    public T call(Callable<T> callable) throws Exception {
        return timeLimiter.callWithTimeout(callable, duration, timeUnit);
    }
}
